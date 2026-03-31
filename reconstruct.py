#!/usr/bin/env python3
"""reconstruct.py – Decode a noise latent back to a reconstructed image.

Loads a noise latent produced by invert.py and runs the DDIM (or Heun-DDIM)
reconstruction pass, optionally comparing against the original image to
compute pixel-level and perceptual quality metrics.

Usage
-----
    # Basic reconstruction
    python reconstruct.py --latent latent.npz --output recon.png

    # With quality metrics (requires the original image)
    python reconstruct.py --latent latent.npz --original photo.jpg --output recon.png

    # Both methods side-by-side from pre-computed latents
    python reconstruct.py \
        --latent-ddim latents/photo_ddim.npz \
        --latent-heun latents/photo_heun.npz \
        --original photo.jpg \
        --output comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from heun_ddim.engine import BenchmarkConfig, SDInversionEngine, save_uint8_image
from heun_ddim.metrics import MetricCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a noise latent back to a reconstructed image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Single-method mode
    parser.add_argument("--latent", type=Path, default=None,
                        help="Path to .npz latent file (single-method mode).")
    parser.add_argument("--method", choices=["ddim", "heun"], default="heun",
                        help="Method that produced the latent (default: heun).")
    # Two-method comparison mode
    parser.add_argument("--latent-ddim", type=Path, default=None,
                        help="DDIM latent .npz (comparison mode).")
    parser.add_argument("--latent-heun", type=Path, default=None,
                        help="Heun-DDIM latent .npz (comparison mode).")
    # Shared
    parser.add_argument("--original", type=Path, default=None,
                        help="Original image for metric computation (optional).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output image path (.png).")
    parser.add_argument("--prompt", type=str, default="a photo",
                        help="Text prompt used during inversion (default: 'a photo').")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of DDIM scheduler steps (default: 50).")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Image size used during inversion (default: 512).")
    return parser.parse_args()


def load_latent(path: Path):
    data = np.load(path)
    import torch
    return torch.from_numpy(data["latent"])


def print_metrics(method_label: str, metrics: dict) -> None:
    print(f"  {method_label:<12}  PSNR {metrics['psnr']:.2f} dB"
          f"  SSIM {metrics['ssim']:.4f}"
          f"  CLIP {metrics['clip_similarity']:.4f}"
          f"  MSE {metrics['mse']:.1f}")


def main() -> None:
    args = parse_args()

    comparison_mode = args.latent_ddim is not None and args.latent_heun is not None
    single_mode     = args.latent is not None

    if not comparison_mode and not single_mode:
        raise ValueError("Provide either --latent or both --latent-ddim and --latent-heun.")

    config = BenchmarkConfig(
        image_size=args.image_size,
        num_steps=args.steps,
        prompt=args.prompt,
    )
    engine = SDInversionEngine(config)

    original_arr = None
    if args.original is not None:
        original_arr = np.array(Image.open(args.original).convert("RGB").resize(
            (args.image_size, args.image_size), Image.LANCZOS
        ))
        calc = MetricCalculator(config.device, fid_keys=[])

    if single_mode:
        latent = load_latent(args.latent)
        print(f"Reconstructing with {args.method.upper()} …")
        recon = engine.decode_latent(latent, method=args.method, prompt=args.prompt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_uint8_image(recon, args.output)
        print(f"  Saved → {args.output}")
        if original_arr is not None:
            print("\nQuality metrics:")
            print_metrics(args.method.upper(), calc.compute_pair_metrics(original_arr, recon))

    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        latent_ddim = load_latent(args.latent_ddim)
        latent_heun = load_latent(args.latent_heun)

        print("Reconstructing DDIM …")
        recon_ddim = engine.decode_latent(latent_ddim, method="ddim", prompt=args.prompt)
        print("Reconstructing Heun-DDIM …")
        recon_heun = engine.decode_latent(latent_heun, method="heun", prompt=args.prompt)

        if original_arr is not None:
            print("\nQuality metrics:")
            m_ddim = calc.compute_pair_metrics(original_arr, recon_ddim)
            m_heun = calc.compute_pair_metrics(original_arr, recon_heun)
            print_metrics("DDIM",      m_ddim)
            print_metrics("Heun-DDIM", m_heun)
            panels = [
                ("Original",  original_arr, None),
                (f"DDIM\nPSNR {m_ddim['psnr']:.2f} dB  SSIM {m_ddim['ssim']:.4f}",
                 recon_ddim, "#4C78A8"),
                (f"Heun-DDIM\nPSNR {m_heun['psnr']:.2f} dB  SSIM {m_heun['ssim']:.4f}",
                 recon_heun, "#F58518"),
            ]
        else:
            panels = [
                ("DDIM",      recon_ddim, "#4C78A8"),
                ("Heun-DDIM", recon_heun, "#F58518"),
            ]

        fig = plt.figure(figsize=(5 * len(panels), 6))
        gs  = gridspec.GridSpec(1, len(panels), wspace=0.04)
        for i, (title, img, color) in enumerate(panels):
            ax = fig.add_subplot(gs[i])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            kw = dict(fontsize=12, fontweight="bold", pad=6)
            if color:
                kw["color"] = color
            ax.set_title(title, **kw)
            for spine in ax.spines.values():
                spine.set_visible(False)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved comparison strip → {args.output}")


if __name__ == "__main__":
    main()
