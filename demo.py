#!/usr/bin/env python3
"""demo.py – Run Heun-DDIM vs DDIM on the six bundled example images.

Reproduces the comparison strips shown in the README using the six COCO images
included in assets/examples/.  Saves side-by-side strips to demo_output/ and
prints a metric summary table.

Usage
-----
    python demo.py
    python demo.py --output-dir my_results/ --steps 50
    python demo.py --example 1          # run only example_01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from heun_ddim.engine import BenchmarkConfig, SDInversionEngine
from heun_ddim.metrics import MetricCalculator

REPO_ROOT    = Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "assets" / "examples"
METHOD_COLORS = {"ddim": "#4C78A8", "heun": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: run Heun-DDIM vs DDIM on the six bundled example images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"),
                        help="Directory for comparison strips (default: demo_output/).")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM scheduler steps (default: 50).")
    parser.add_argument("--prompt", type=str, default="a photo",
                        help="Text conditioning prompt (default: 'a photo').")
    parser.add_argument("--example", type=int, default=None,
                        help="Run only this example number (1-6). Omit to run all.")
    return parser.parse_args()


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS))


def save_strip(
    original: np.ndarray,
    recon_ddim: np.ndarray,
    recon_heun: np.ndarray,
    m_ddim: dict,
    m_heun: dict,
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.04)
    panels = [
        ("Original", original, None),
        (f"DDIM\nPSNR {m_ddim['psnr']:.2f} dB  SSIM {m_ddim['ssim']:.4f}",
         recon_ddim, METHOD_COLORS["ddim"]),
        (f"Heun-DDIM\nPSNR {m_heun['psnr']:.2f} dB  SSIM {m_heun['ssim']:.4f}",
         recon_heun, METHOD_COLORS["heun"]),
    ]
    for i, (label, img, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        kw = dict(fontsize=12, fontweight="bold", pad=6)
        if color:
            kw["color"] = color
        ax.set_title(label, **kw)
        for spine in ax.spines.values():
            spine.set_visible(False)
    delta = m_heun["psnr"] - m_ddim["psnr"]
    fig.suptitle(f"{title}  |  Heun-DDIM PSNR gain: {delta:+.2f} dB", fontsize=13, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Discover example images
    example_paths = sorted(EXAMPLES_DIR.glob("example_*.png"))
    if not example_paths:
        raise FileNotFoundError(
            f"No example images found in {EXAMPLES_DIR}.\n"
            "Make sure you cloned the full repo including assets/examples/."
        )
    if args.example is not None:
        example_paths = [p for p in example_paths if p.stem == f"example_{args.example:02d}"]
        if not example_paths:
            raise ValueError(f"example_{args.example:02d}.png not found in {EXAMPLES_DIR}")

    config = BenchmarkConfig(num_steps=args.steps, prompt=args.prompt)
    engine = SDInversionEngine(config)
    calc   = MetricCalculator(config.device, fid_keys=[])

    print(f"Device : {config.device}")
    print(f"Steps  : {args.steps}")
    print(f"Prompt : {args.prompt!r}")
    print(f"Images : {len(example_paths)}")
    print()

    rows = []
    sep  = "-" * 72
    print(f"{'Example':<12} {'Method':<12} {'PSNR':>8} {'SSIM':>8} {'CLIP':>8}")
    print(sep)

    for img_path in example_paths:
        name = img_path.stem  # e.g. "example_01"
        original = load_rgb(img_path, config.image_size)
        img_tensor, _ = engine.preprocess_image(img_path)

        results = {}
        for method in ("ddim", "heun"):
            run            = engine.run_method(img_tensor, method, prompt=args.prompt)
            recon          = run["reconstruction_image"]
            m              = calc.compute_pair_metrics(original, recon)
            results[method] = {"recon": recon, "metrics": m}
            label = "DDIM" if method == "ddim" else "Heun-DDIM"
            print(f"{name:<12} {label:<12} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {m['clip_similarity']:>8.4f}")
            rows.append({"example": name, "method": label, **m})

        save_strip(
            original,
            results["ddim"]["recon"],
            results["heun"]["recon"],
            results["ddim"]["metrics"],
            results["heun"]["metrics"],
            args.output_dir / f"{name}.png",
            name,
        )
        delta = results["heun"]["metrics"]["psnr"] - results["ddim"]["metrics"]["psnr"]
        print(f"  → saved {args.output_dir / name}.png  (Δ PSNR {delta:+.2f} dB)")
        print()

    print(sep)
    print(f"Strips saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
