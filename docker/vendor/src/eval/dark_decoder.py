"""src/eval/dark_decoder.py — DARK sub-pixel heatmap decoding (Zhang CVPR'20).

Inference-only refinement of hard argmax. Takes the same heatmap tensor
the organizer baseline decoder consumes, returns the same normalized
[x_norm, y_norm] coords (in [0, 1] over the heatmap grid), but with
sub-pixel refinement via Taylor expansion around the argmax cell.

Math (Zhang 2020 §3.2):
  1. Smooth heatmap with Gaussian (we use the training σ in cell units).
  2. Take log to linearize the Gaussian into a paraboloid.
  3. At argmax cell (px, py), compute gradient g and Hessian H of log(hm).
  4. Sub-pixel offset = -H^-1 · g. This is the Newton step toward the
     paraboloid peak.
  5. Refined coord = (px, py) + offset, clipped to ±0.5 cell.
  6. Normalize by (w-1, h-1) so output matches the organizer baseline
     `decode_heatmaps_to_normalized_coords` contract bit-for-bit when
     DARK falls back to argmax (boundary cells / non-PD Hessian).

Fallback to argmax when:
  - Peak lies on the heatmap border (no full 3×3 stencil).
  - Hessian is not negative-definite (peak is saddle/plateau).
  - Determinant near 0 (numerically unstable inverse).

Contract (verified vs baseline/utils.py:28-39):
  - Same input shape: (B, K, H, W) torch tensor.
  - Same output shape: (B, 2K) torch tensor of normalized coords.
  - argmax-decoder output exactly equal when fallback fires for all peaks
    (sanity-checkable in unit tests).

Inference-only; NOT differentiable (scipy gaussian_filter on CPU). For
training-time differentiable variant use DSNT (different technique).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_EPS = 1e-10
# DARK paper recommends smoothing σ matched to training σ in cell units.
# Our train σ = 1.8 (in cells of a 64×64 heatmap). Same value here.
_SMOOTH_SIGMA_DEFAULT = 1.8


def dark_decode_2d(
    heatmap_2d: np.ndarray,
    smooth_sigma: float = _SMOOTH_SIGMA_DEFAULT,
) -> tuple[float, float]:
    """Refine argmax peak of a 2D heatmap to sub-pixel.

    Args
    ----
    heatmap_2d: (H, W) float numpy array. Values can be in [0, 1] (sigmoid
        output) or unbounded logits — only the SHAPE of the peak matters.
    smooth_sigma: Gaussian smoothing σ in cell units. Default 1.8 = matches
        our training σ (HEATMAP_SIGMA in train_m0.py).

    Returns
    -------
    (x_subpixel, y_subpixel) in cell-grid coordinates [0, W-1] × [0, H-1].
    """
    hm = np.asarray(heatmap_2d, dtype=np.float64)
    h, w = hm.shape

    # Step 1 — smooth before peak fit (DARK §3.2; reduces noise sensitivity)
    blurred = gaussian_filter(hm, sigma=smooth_sigma)

    # Step 2 — find argmax cell
    flat_idx = int(np.argmax(blurred))
    py = flat_idx // w
    px = flat_idx % w

    # Boundary check → fall back to argmax if 3x3 stencil won't fit
    if px < 1 or px >= w - 1 or py < 1 or py >= h - 1:
        return float(px), float(py)

    # Step 3 — log-space (Gaussian → paraboloid)
    log_hm = np.log(np.maximum(blurred, _EPS))

    # Step 4a — gradient at (px, py)
    dx = (log_hm[py, px + 1] - log_hm[py, px - 1]) / 2.0
    dy = (log_hm[py + 1, px] - log_hm[py - 1, px]) / 2.0

    # Step 4b — Hessian at (px, py)
    dxx = log_hm[py, px + 1] - 2.0 * log_hm[py, px] + log_hm[py, px - 1]
    dyy = log_hm[py + 1, px] - 2.0 * log_hm[py, px] + log_hm[py - 1, px]
    dxy = (
        log_hm[py + 1, px + 1]
        - log_hm[py + 1, px - 1]
        - log_hm[py - 1, px + 1]
        + log_hm[py - 1, px - 1]
    ) / 4.0

    # Step 4c — peak validity: Hessian must be negative-definite (true max)
    det = dxx * dyy - dxy * dxy
    if dxx >= 0.0 or dyy >= 0.0 or det <= _EPS:
        return float(px), float(py)

    # Step 5 — Newton step toward paraboloid peak: offset = -H^-1 · g
    # 2x2 inverse closed-form
    inv_det = 1.0 / det
    H_inv = np.array(
        [[dyy * inv_det, -dxy * inv_det], [-dxy * inv_det, dxx * inv_det]],
        dtype=np.float64,
    )
    offset = -H_inv @ np.array([dx, dy], dtype=np.float64)

    # Clip to ±0.5 cell — refinement should stay within argmax cell
    offset = np.clip(offset, -0.5, 0.5)

    return float(px + offset[0]), float(py + offset[1])


def decode_heatmaps_dark(heatmaps: torch.Tensor, smooth_sigma: float = _SMOOTH_SIGMA_DEFAULT) -> torch.Tensor:
    """Drop-in replacement for baseline.utils.decode_heatmaps_to_normalized_coords.

    Args
    ----
    heatmaps: (B, K, H, W) torch tensor.
    smooth_sigma: cell-unit σ for the Gaussian pre-smoothing.

    Returns
    -------
    (B, 2K) torch tensor of normalized coords. K landmarks → flat
    [x1, y1, x2, y2, ...] order, same as the baseline argmax decoder.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must be (B, K, H, W); got {tuple(heatmaps.shape)}")
    bsz, num_points, h, w = heatmaps.shape
    hm_np = heatmaps.detach().cpu().numpy()

    out = np.zeros((bsz, num_points * 2), dtype=np.float32)
    norm_x = max(float(w - 1), 1.0)
    norm_y = max(float(h - 1), 1.0)
    for b in range(bsz):
        for k in range(num_points):
            xs, ys = dark_decode_2d(hm_np[b, k], smooth_sigma=smooth_sigma)
            out[b, 2 * k + 0] = xs / norm_x
            out[b, 2 * k + 1] = ys / norm_y

    return torch.from_numpy(out).to(heatmaps.device)


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Reference vanilla argmax decoder — bit-identical to
    baseline/utils.py::decode_heatmaps_to_normalized_coords. Included here
    so a single eval loop can call both decoders on the same forward pass
    without importing organizer code."""
    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must be (B, K, H, W); got {tuple(heatmaps.shape)}")
    bsz, num_points, h, w = heatmaps.shape
    flat_idx = heatmaps.view(bsz, num_points, -1).argmax(dim=-1)
    ys = torch.div(flat_idx, w, rounding_mode="floor").float()
    xs = (flat_idx % w).float()
    x_norm = xs / max(float(w - 1), 1.0)
    y_norm = ys / max(float(h - 1), 1.0)
    return torch.stack([x_norm, y_norm], dim=-1).reshape(bsz, num_points * 2)
