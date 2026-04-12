# Heun-DDIM: Second-Order Corrected DDIM Inversion for Stable Diffusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: SD 1.5](https://img.shields.io/badge/model-SD%201.5-orange.svg)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

**Heun-DDIM** applies a predictor-corrector correction to DDIM inversion,
reducing round-trip reconstruction error with no changes to the model weights.
Evaluated on **5,000 COCO images** at 50 steps, it raises mean PSNR from
**22.1 → 25.1 dB**, lowers LPIPS from **0.224 → 0.081**, raises CLIP
similarity from **0.861 → 0.974**, and lowers FID from **15.6 → 3.4**
under true no-text conditioning.

---

## Teaser

<p align="center"><img src="assets/comparisons/example_01.png"/></p>
<p align="center"><em>Left: original image.  Center: DDIM reconstruction.  Right: Heun-DDIM reconstruction.</em></p>

---

## Method

### The problem with DDIM inversion

DDIM inversion reverses the deterministic DDIM sampling process to map a real
image back into the noise space of the diffusion model.  The standard approach
treats the UNet score estimate at timestep *t* as constant over the entire
integration step — a first-order (Euler) approximation that accumulates
truncation error across all 50 steps.  The resulting noise code cannot be
perfectly decoded back to the original image.

### Heun correction

We apply the classical **Heun predictor-corrector** scheme to each inversion step:

```
# Predictor (standard DDIM step)
eps_t    = unet(z_t, t)
z0_pred  = (z_t - sqrt(1-a_t) * eps_t) / sqrt(a_t)
z_next   = sqrt(a_{t+1}) * z0_pred + sqrt(1-a_{t+1}) * eps_t

# Corrector (re-evaluate at predicted state, average scores)
eps_next = unet(z_next, t+1)
eps_avg  = (eps_t + eps_next) / 2

z0_corr  = (z_t - sqrt(1-a_t) * eps_avg) / sqrt(a_t)
z_next   = sqrt(a_{t+1}) * z0_corr + sqrt(1-a_{t+1}) * eps_avg
```

The averaged score is a second-order approximation of the true ODE velocity,
halving the local truncation error at the cost of one extra UNet forward pass
per step.  The same correction is applied symmetrically during reconstruction.

---

## Results

All experiments use Stable Diffusion 1.5, 512×512, 50 DDIM steps, 5,000 COCO 2017
images.  LPIPS is computed with AlexNet, and FID is recomputed separately with
InceptionV3 pool_3 features (feature=2048) using `scripts/recompute_fid.py`.
Three prompt conditions are reported separately.

### No-text inversion  (empty prompt)

| Method    |   N   |  MSE ↓  | PSNR ↑ | SSIM ↑ | CLIP ↑ | LPIPS ↓ | FID ↓ | Total (s) |
|-----------|------:|--------:|-------:|-------:|-------:|--------:|------:|----------:|
| DDIM      | 5 000 | 536.81  | 22.11  | 0.6364 | 0.8614 | 0.2243  | 15.59 | 2.99      |
| Heun-DDIM | 5 000 | **311.58** | **25.10** | **0.7201** | **0.9737** | **0.0814** | **3.37** | 5.86 |

Δ PSNR = **+3.00 dB**  ·  Δ SSIM = **+0.084**  ·  LPIPS reduction = **64 %**  ·  FID reduction = **78 %**

<!-- <p align="center"><img src="assets/plots/metrics_table_no_text.png"/></p>
<p align="center"><em>Metrics — No Text Prompt</em></p> -->

<p align="center"><img src="assets/plots/metrics_histograms_no_text.png"/></p>
<p align="center"><em>Per-image distributions — No Text Prompt</em></p>

---

### Fixed-prompt inversion  (`"a photo"` prompt)

| Method    |   N   |  MSE ↓  | PSNR ↑ | SSIM ↑ | CLIP ↑ | LPIPS ↓ | FID ↓ | Total (s) |
|-----------|------:|--------:|-------:|-------:|-------:|--------:|------:|----------:|
| DDIM      | 5 000 | 532.43  | 22.11  | 0.6367 | 0.8655 | 0.2235  | 15.79 | 2.99      |
| Heun-DDIM | 5 000 | **311.13** | **25.10** | **0.7202** | **0.9739** | **0.0812** | **3.37** | 5.86 |

Δ PSNR = **+2.99 dB**  ·  Δ SSIM = **+0.084**  ·  LPIPS reduction = **64 %**  ·  FID reduction = **79 %**

<!-- <p align="center"><img src="assets/plots/metrics_table_fixed.png"/></p>
<p align="center"><em>Metrics — Fixed Prompt</em></p> -->

<p align="center"><img src="assets/plots/metrics_histograms_fixed.png"/></p>
<p align="center"><em>Per-image distributions — Fixed Prompt</em></p>

---

### With-text inversion  (COCO category-label prompt)

| Method    |   N   |  MSE ↓  | PSNR ↑ | SSIM ↑ | CLIP ↑ | LPIPS ↓ | FID ↓ | Total (s) |
|-----------|------:|--------:|-------:|-------:|-------:|--------:|------:|----------:|
| DDIM      | 5 000 | 507.16  | 22.39  | 0.6457 | 0.8964 | 0.2092  | 11.69 | 2.99      |
| Heun-DDIM | 5 000 | **311.18** | **25.13** | **0.7204** | **0.9751** | **0.0808** | **3.27** | 5.87 |

Δ PSNR = **+2.74 dB**  ·  Δ SSIM = **+0.075**  ·  LPIPS reduction = **61 %**  ·  FID reduction = **72 %**

<!-- <p align="center"><img src="assets/plots/metrics_table_with_text.png"/></p>
<p align="center"><em>Metrics — Category-Label Prompt</em></p> -->

<p align="center"><img src="assets/plots/metrics_histograms_with_text.png"/></p>
<p align="center"><em>Per-image distributions — Category-Label Prompt</em></p>

> Category-label prompts give DDIM a small boost over no-text conditioning
> (22.11 → 22.39 dB), but
> Heun-DDIM's advantage is consistent across all three conditions.

### Visual comparisons

Top-6 no-text images by Heun-DDIM PSNR improvement over DDIM.

<p align="center"><img src="assets/comparisons/example_01.png"/></p>
<p align="center"><img src="assets/comparisons/example_02.png"/></p>
<p align="center"><img src="assets/comparisons/example_03.png"/></p>
<p align="center"><img src="assets/comparisons/example_04.png"/></p>
<p align="center"><img src="assets/comparisons/example_05.png"/></p>
<p align="center"><img src="assets/comparisons/example_06.png"/></p>
<p align="center"><em>Left: original.  Center: DDIM.  Right: Heun-DDIM.</em></p>

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/OmerBibi/heun-ddim
cd heun-ddim
pip install -r requirements.txt
```

### 2. Try the demo

Run both methods on the six bundled COCO example images and save comparison strips:

```bash
python demo.py
```

Results are written to `demo_output/`.  Each strip shows the original image,
the DDIM reconstruction, and the Heun-DDIM reconstruction side-by-side with
PSNR / SSIM annotations.  A metric summary table is printed to stdout:

```
Example      Method       PSNR     SSIM     CLIP
------------------------------------------------------------------------
example_01   DDIM         19.23    0.6102   0.8541
example_01   Heun-DDIM    38.99    0.9821   0.9991
  → saved demo_output/example_01.png  (Δ PSNR +19.76 dB)
...
```

---

## Scripts

### `invert.py` — invert an image to its noise latent

```bash
# Heun-DDIM inversion (recommended)
python invert.py --image photo.jpg --output latent.npz --method heun

# Standard DDIM inversion
python invert.py --image photo.jpg --output latent.npz --method ddim

# Both methods saved to a directory
python invert.py --image photo.jpg --output-dir latents/ --method both
```

The noise latent is saved as a compressed `.npz` file.  Pass `--prompt` to
use a custom text conditioning string (default: `"a photo"`).

---

### `reconstruct.py` — decode a latent back to an image

```bash
# Single latent → reconstructed image
python reconstruct.py --latent latent.npz --output recon.png

# With quality metrics against the original
python reconstruct.py \
    --latent   latent.npz \
    --original photo.jpg \
    --output   recon.png

# Side-by-side comparison strip from two latents
python reconstruct.py \
    --latent-ddim  latents/photo_ddim.npz \
    --latent-heun  latents/photo_heun.npz \
    --original     photo.jpg \
    --output       comparison.png
```

When `--original` is provided, PSNR / SSIM / CLIP scores are printed:

```
Quality metrics:
  DDIM          PSNR 22.10 dB  SSIM 0.6385  CLIP 0.8648  MSE 530.8
  Heun-DDIM     PSNR 25.14 dB  SSIM 0.7230  CLIP 0.9735  MSE 303.3
```

---

### `demo.py` — reproduce the README comparison figures

```bash
# Run all six bundled examples
python demo.py

# Run a single example
python demo.py --example 3

# Custom output directory and step count
python demo.py --output-dir results/ --steps 50
```

---

## Full inversion + reconstruction workflow

```bash
# Step 1 — invert with both methods
python invert.py \
    --image      photo.jpg \
    --output-dir latents/ \
    --method     both \
    --steps      50 \
    --prompt     "a photo"

# Step 2 — reconstruct and compare
python reconstruct.py \
    --latent-ddim  latents/photo_ddim.npz \
    --latent-heun  latents/photo_heun.npz \
    --original     photo.jpg \
    --output       comparison.png
```

For benchmark runs, `scripts/run_benchmark.py` now also supports a true
empty-conditioning mode:

```bash
python scripts/run_benchmark.py \
    --manifest /path/to/manifest.csv \
    --output-root runs/coco_4k_no_text \
    --prompt-source no_text \
    --num-steps 50 \
    --image-size 512
```

---

## Citation

```bibtex
@misc{heun-ddim-2026,
  title  = {Heun-DDIM: Second-Order Corrected DDIM Inversion for Stable Diffusion},
  author = {Bibi, Omer and Adar, Yotam},
  year   = {2026},
  url    = {https://github.com/OmerBibi/heun-ddim},
}
```
