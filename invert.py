#!/usr/bin/env python3
"""invert.py – Invert a single image to its diffusion noise latent.

Runs DDIM or Heun-DDIM inversion on a JPEG/PNG image and saves the
resulting noise latent as a compressed NumPy archive (.npz).  The latent
can later be decoded back to an image with reconstruct.py.

Usage
-----
    # Heun-DDIM inversion (recommended)
    python invert.py --image photo.jpg --output latent.npz --method heun

    # Standard DDIM inversion
    python invert.py --image photo.jpg --output latent.npz --method ddim

    # Both methods, saved to separate files
    python invert.py --image photo.jpg --output-dir latents/ --method both
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from heun_ddim.engine import BenchmarkConfig, SDInversionEngine, serialize_tensor_to_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invert an image to its diffusion noise latent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Input image path (.jpg or .png).")
    parser.add_argument("--method", choices=["ddim", "heun", "both"], default="heun",
                        help="Inversion method (default: heun).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .npz path. Used when --method is ddim or heun.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Used when --method is both.")
    parser.add_argument("--prompt", type=str, default="a photo",
                        help="Text prompt for inversion conditioning (default: 'a photo').")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of DDIM scheduler steps (default: 50).")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Resize/crop size in pixels (default: 512).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    config = BenchmarkConfig(
        image_size=args.image_size,
        num_steps=args.steps,
        prompt=args.prompt,
    )
    engine = SDInversionEngine(config)

    print(f"Loading model … (device: {config.device})")
    image_tensor, _ = engine.preprocess_image(args.image)

    methods = ["ddim", "heun"] if args.method == "both" else [args.method]

    for method in methods:
        print(f"\nRunning {method.upper()} inversion ({args.steps} steps) …")
        t0 = time.perf_counter()
        result = engine.run_method(image_tensor, method, prompt=args.prompt)
        elapsed = time.perf_counter() - t0

        if args.method == "both":
            out_dir = args.output_dir or Path(".")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.image.stem}_{method}.npz"
        else:
            out_path = args.output or Path(f"{args.image.stem}_{method}.npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)

        serialize_tensor_to_npz(result["noise_latent"], out_path)
        print(f"  Saved latent → {out_path}  ({elapsed:.2f} s)")

    print("\nDone.  Use reconstruct.py to decode the latent back to an image.")


if __name__ == "__main__":
    main()
