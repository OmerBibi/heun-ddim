"""metrics.py – Image quality metrics and FID computation for heun_ddim.

Supports per-image metrics (MSE, PSNR, SSIM, CLIP similarity) and
accumulates per-method FID scores over the full evaluation set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPModel, CLIPProcessor


CLIP_ID = "openai/clip-vit-base-patch32"


class MetricCalculator:
    """Computes pixel-level and perceptual metrics; accumulates FID per method."""

    def __init__(self, device: str, fid_keys: List[str] | None = None):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
        self.lpips_model = lpips.LPIPS(net="alex").to(device)
        self.lpips_model.eval()
        if fid_keys is None:
            fid_keys = ["ddim", "heun"]
        self.fid_by_method = {key: FrechetInceptionDistance(feature=2048).to(device) for key in fid_keys}

    def compute_clip_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        inputs = self.clip_processor(images=[img_a, img_b], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        similarity = torch.matmul(features[0], features[1])
        return float(similarity.item())

    def compute_pair_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        original_tensor = (
            torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        )
        reconstructed_tensor = (
            torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        )
        with torch.no_grad():
            lpips_value = self.lpips_model(original_tensor, reconstructed_tensor, normalize=True)
        return {
            "mse": float(mean_squared_error(original, reconstructed)),
            "psnr": float(psnr(original, reconstructed, data_range=255)),
            "ssim": float(ssim(original, reconstructed, data_range=255, channel_axis=2, win_size=7)),
            "clip_similarity": self.compute_clip_similarity(original, reconstructed),
            "lpips": float(lpips_value.item()),
        }

    def update_fid(self, method: str, original: np.ndarray, reconstructed: np.ndarray) -> None:
        real = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.uint8)
        fake = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.uint8)
        self.fid_by_method[method].update(real, real=True)
        self.fid_by_method[method].update(fake, real=False)

    def finalize_fid(self) -> Dict[str, float]:
        return {method: float(metric.compute().item()) for method, metric in self.fid_by_method.items()}


def write_summary(results_df: pd.DataFrame, fid_scores: Dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    group_cols = ["method"]
    if "prompt_mode" in results_df.columns:
        group_cols.append("prompt_mode")

    for group_key, method_df in results_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            method = str(group_key[0])
            prompt_mode = str(group_key[1])
        else:
            method = str(group_key)
            prompt_mode = None
        fid_key = method if prompt_mode is None else f"{prompt_mode}:{method}"
        row = {
            "method": method,
            "num_samples": int(len(method_df)),
            "mean_mse": float(method_df["mse"].mean()),
            "mean_psnr": float(method_df["psnr"].mean()),
            "mean_ssim": float(method_df["ssim"].mean()),
            "mean_clip_similarity": float(method_df["clip_similarity"].mean()),
            "mean_lpips": float(method_df["lpips"].mean()),
            "mean_inversion_seconds": float(method_df["inversion_seconds"].mean()),
            "mean_reconstruction_seconds": float(method_df["reconstruction_seconds"].mean()),
            "mean_total_seconds": float(method_df["total_seconds"].mean()),
            "fid": float(fid_scores.get(fid_key, float("nan"))),
        }
        if prompt_mode is not None:
            row["prompt_mode"] = prompt_mode
        summary_rows.append(row)

    sort_cols = ["method"]
    if summary_rows and "prompt_mode" in summary_rows[0]:
        sort_cols = ["prompt_mode", "method"]
    summary_df = pd.DataFrame(summary_rows).sort_values(sort_cols)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
