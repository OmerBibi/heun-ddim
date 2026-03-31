"""engine.py – SD 1.5 inversion engine supporting DDIM and Heun-corrected DDIM.

DDIM inversion approximates the reverse ODE by treating the score estimate at
the *current* noisy state as constant over the entire step.  This first-order
Euler-like approximation accumulates truncation error that prevents exact
round-trip reconstruction.

Heun-DDIM corrects for this by performing a predictor-corrector step:
  1. Predict the noise estimate at the next state with a standard DDIM step.
  2. Re-evaluate the score at that predicted state.
  3. Average the two estimates and redo the step with the better score.
This halves the local truncation error (second-order accuracy) and
substantially improves PSNR / SSIM on real photographs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


LATENT_SCALE = 0.18215
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_PROMPT = "a photo"

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "ddim": "DDIM",
    "heun": "Heun-DDIM",
}


@dataclass
class BenchmarkConfig:
    model_id: str = DEFAULT_MODEL_ID
    image_size: int = 512
    num_steps: int = 50
    prompt: str = DEFAULT_PROMPT
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.device == "cuda" else torch.float32


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class SDInversionEngine:
    """Loads SD 1.5 once and exposes DDIM and Heun-DDIM inversion/reconstruction."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.torch_dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            config.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.model_id,
            subfolder="unet",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()

        self._preprocess = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=Image.Resampling.LANCZOS),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.prompt_embedding = self.get_text_embedding(config.prompt)

    def _condition_embedding(self, prompt: str | None, batch_size: int) -> torch.Tensor:
        cond = self.prompt_embedding if prompt is None else self.get_text_embedding(prompt)
        if cond.shape[0] == 1:
            cond = cond.expand(batch_size, -1, -1)
        return cond

    @torch.no_grad()
    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    def preprocess_image(self, image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        transformed = self._preprocess(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        preview = transformed.squeeze(0).detach().cpu().float()
        preview = ((preview * 0.5) + 0.5).clamp(0, 1)
        preview = (preview.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return transformed, preview

    @torch.no_grad()
    def encode_to_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(image_tensor).latent_dist
        return posterior.sample() * LATENT_SCALE

    @torch.no_grad()
    def invert_ddim(self, z_latents: torch.Tensor, prompt: str | None = None) -> torch.Tensor:
        """Standard DDIM inversion: first-order Euler integration in reverse time."""
        self.scheduler.set_timesteps(self.config.num_steps, device=self.device)
        timesteps = list(reversed(self.scheduler.timesteps))
        z = z_latents.clone()
        cond = self._condition_embedding(prompt, z.shape[0])

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            eps = self.unet(z, t_curr, encoder_hidden_states=cond).sample
            at = self.scheduler.alphas_cumprod[t_curr]
            at_next = self.scheduler.alphas_cumprod[t_next]
            # Predict clean image from current noisy state, then re-noise to t_next
            z0 = (z - (1 - at) ** 0.5 * eps) / (at**0.5)
            z = (at_next**0.5) * z0 + ((1 - at_next) ** 0.5) * eps
        return z

    @torch.no_grad()
    def invert_heun(self, z_latents: torch.Tensor, prompt: str | None = None) -> torch.Tensor:
        """Heun-corrected DDIM inversion: predictor-corrector for second-order accuracy.

        After the Euler predictor step we get a candidate z_{t+1}.  We then
        re-evaluate the UNet score at that candidate and average the two
        score estimates before computing the final step.  The averaged score
        is a much better approximation of the true ODE velocity, reducing
        round-trip reconstruction error.
        """
        self.scheduler.set_timesteps(self.config.num_steps, device=self.device)
        timesteps = list(reversed(self.scheduler.timesteps))
        z = z_latents.clone()
        cond = self._condition_embedding(prompt, z.shape[0])

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            # Predictor: standard DDIM step
            eps = self.unet(z, t_curr, encoder_hidden_states=cond).sample
            at = self.scheduler.alphas_cumprod[t_curr]
            at_next = self.scheduler.alphas_cumprod[t_next]
            z0_t = (z - (1 - at) ** 0.5 * eps) / (at**0.5)
            z_next_euler = (at_next**0.5) * z0_t + ((1 - at_next) ** 0.5) * eps
            # Corrector: re-evaluate score at predicted state, then average
            eps_next = self.unet(z_next_euler, t_next, encoder_hidden_states=cond).sample
            eps_avg = (eps + eps_next) / 2.0
            z0_better = (z - (1 - at) ** 0.5 * eps_avg) / (at**0.5)
            z = (at_next**0.5) * z0_better + ((1 - at_next) ** 0.5) * eps_avg
        return z

    @torch.no_grad()
    def reconstruct_ddim(self, noise_latents: torch.Tensor, prompt: str | None = None) -> torch.Tensor:
        """Standard DDIM denoising (forward time)."""
        self.scheduler.set_timesteps(self.config.num_steps, device=self.device)
        timesteps = list(self.scheduler.timesteps)
        z = noise_latents.clone()
        cond = self._condition_embedding(prompt, z.shape[0])

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            eps = self.unet(z, t_curr, encoder_hidden_states=cond).sample
            at = self.scheduler.alphas_cumprod[t_curr]
            at_next = self.scheduler.alphas_cumprod[t_next]
            z0 = (z - (1 - at) ** 0.5 * eps) / (at**0.5)
            z = (at_next**0.5) * z0 + ((1 - at_next) ** 0.5) * eps
        return z

    @torch.no_grad()
    def reconstruct_heun(self, noise_latents: torch.Tensor, prompt: str | None = None) -> torch.Tensor:
        """Heun-corrected DDIM denoising (forward time)."""
        self.scheduler.set_timesteps(self.config.num_steps, device=self.device)
        timesteps = list(self.scheduler.timesteps)
        z = noise_latents.clone()
        cond = self._condition_embedding(prompt, z.shape[0])

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            eps = self.unet(z, t_curr, encoder_hidden_states=cond).sample
            at = self.scheduler.alphas_cumprod[t_curr]
            at_next = self.scheduler.alphas_cumprod[t_next]
            z0_t = (z - (1 - at) ** 0.5 * eps) / (at**0.5)
            z_prev_euler = (at_next**0.5) * z0_t + ((1 - at_next) ** 0.5) * eps
            eps_next = self.unet(z_prev_euler, t_next, encoder_hidden_states=cond).sample
            eps_avg = (eps + eps_next) / 2.0
            z0_better = (z - (1 - at) ** 0.5 * eps_avg) / (at**0.5)
            z = (at_next**0.5) * z0_better + ((1 - at_next) ** 0.5) * eps_avg
        return z

    @torch.no_grad()
    def decode_latents_to_uint8(self, latents: torch.Tensor) -> np.ndarray:
        latents = (1.0 / LATENT_SCALE) * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype(np.uint8)
        return image[0]

    def run_method(self, image_tensor: torch.Tensor, method: str, prompt: str | None = None) -> Dict[str, object]:
        if method not in {"ddim", "heun"}:
            raise ValueError(f"Unsupported method: {method!r}.  Choose 'ddim' or 'heun'.")

        invert_fn = self.invert_ddim if method == "ddim" else self.invert_heun
        reconstruct_fn = self.reconstruct_ddim if method == "ddim" else self.reconstruct_heun

        sync_device(self.device)
        start = time.perf_counter()
        z0 = self.encode_to_latent(image_tensor)
        zt = invert_fn(z0, prompt=prompt)
        sync_device(self.device)
        inversion_seconds = time.perf_counter() - start

        sync_device(self.device)
        start = time.perf_counter()
        recon_latent = reconstruct_fn(zt, prompt=prompt)
        recon_image = self.decode_latents_to_uint8(recon_latent)
        sync_device(self.device)
        reconstruction_seconds = time.perf_counter() - start

        return {
            "noise_latent": zt.detach().cpu(),
            "reconstruction_latent": recon_latent.detach().cpu(),
            "reconstruction_image": recon_image,
            "inversion_seconds": inversion_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "total_seconds": inversion_seconds + reconstruction_seconds,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def image_extensions() -> List[str]:
    return [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def list_images(image_root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in image_extensions():
        files.extend(image_root.rglob(f"*{ext}"))
        files.extend(image_root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def save_uint8_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def serialize_tensor_to_npz(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, latent=tensor.squeeze(0).cpu().numpy())
