"""src/training/albu_transforms.py — A.Compose pipelines by preprocess name.

Single source of truth for the train/val albumentations pipelines used by
organizer-math scripts (train_m0_preproc.py and any sibling that vendors
the baseline). Preprocess is selectable by string — same registry idea as
src/training/preprocessing.py but here the function returns an A.Compose
so we can place CLAHE BEFORE photometric aug (the registry function does
the whole pipeline end-to-end which doesn't compose cleanly with aug).

Preprocess names
----------------
- "imagenet_default": byte-equivalent to organizer baseline pipeline
  (Resize → RandomBrightnessContrast(p=0.2) → GaussNoise(p=0.1) →
  ImageNet Normalize → ToTensorV2 for train; Resize → Normalize → ToTensor
  for val). Default — keeps M0 reproducible.

- "clahe_clip": CLAHE(LAB-L, clip=2.0, tile=8x8, p=1.0) → percentile
  clip [1, 99] → organizer photometric aug → ImageNet Normalize →
  ToTensorV2. CLAHE+clip placed BEFORE RandomBrightnessContrast/GaussNoise
  so train and val see the same contrast-equalized base distribution;
  aug only adds variance on top of that base.

- "clahe_clip_sharpen_speckle": clahe_clip + A.Sharpen(p=0.3) +
  A.MultiplicativeNoise(p=0.2) appended to the train-only block. Sharpen
  emphasises edges (anatomy boundaries); MultiplicativeNoise simulates
  speckle pattern variability across US machines. Val pipeline unchanged
  vs clahe_clip.

LANDMINE (do not reintroduce)
-----------------------------
NEVER do per-image percentile *stretch*  `(x - lo) / (hi - lo)`. That was
the 358 px regression (commit 425aaec, fixed e640e8c). The stretch
rescales every image by a content-dependent factor and collapses DINOv2
features. Percentile *clip* (clamp via np.clip, no division) is safe —
matches src/training/preprocessing.py::clahe_clip on the same knobs.

Constants
---------
CLAHE clip_limit=(2.0, 2.0) (tuple-pinned — scalar 2.0 would be silently
randomized to range [1.0, 2.0] in albumentations >= 2.0), tile=8x8;
percentile clip [1, 99]; ImageNet mean/std. Pinned identical to
src/training/preprocessing.py::clahe_clip (cv2.createCLAHE clipLimit=2.0,
deterministic) so both registries are arithmetically consistent.
"""
from __future__ import annotations

from typing import Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# [USFM PATCH BEIT05 2026-06-03] — timm-BEiT-DEFAULT normalization (0.5),
# used as a WRONG-NORM CONTROL for the folder-07 USFM A/B. NOTE (corrected
# 2026-06-03 after reading USFM's own repo): USFM's official downstream
# code normalizes with IMAGENET stats — see openmedlab/USFM
# `usdsgen/data/datasets.py:28-43` (`T.Normalize(IMAGENET_DEFAULT_MEAN/STD)`
# for both train and val). So the PAPER-FAITHFUL norm is ImageNet (the
# existing `clahe_clip_sharpen_speckle` key), NOT 0.5. This 0.5 key is the
# timm-BEiT default kept only as a control to MEASURE the cost of the wrong
# norm — it is NOT "encoder-native" (that earlier framing was wrong). Same
# CLAHE chain as the leader; only the final Normalize differs. 3-channel.
BEIT_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
BEIT_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# [OURS PATCH 1CH BEGIN] — load OURS_MEAN/STD scalar from folder 04
# compute_ours_norm_stats.py output. Used by `clahe_clip_sharpen_speckle_1ch_ours`
# preprocess key (folder 04 nb00 R07-1ch baseline + nb01 iBOT-DAPT downstream).
# Fallback to folder 03 EchoCare labeled-train stats (mean_gray=0.115, std_gray=0.167)
# if 187k corpus stats JSON not yet computed (smoke / dev path).
import json as _json
import os as _os
from pathlib import Path as _Path

_OURS_STATS_PATH = (
    _Path(__file__).resolve().parents[4]
    / "experiments" / "04_dinov2_ultradinoLIKE" / "ours_norm_stats.json"
)
_OURS_FALLBACK_PATH = (
    _Path(__file__).resolve().parents[4]
    / "experiments" / "z_failedtechniques" / "03_echocare_baseline_swinb" / "us_norm_stats.json"
)
if _OURS_STATS_PATH.exists():
    _stats = _json.loads(_OURS_STATS_PATH.read_text())
    OURS_MEAN_GRAY: float = float(_stats["mean_gray"])
    OURS_STD_GRAY: float = float(_stats["std_gray"])
    OURS_STATS_SOURCE: str = str(_OURS_STATS_PATH)
elif _OURS_FALLBACK_PATH.exists():
    _stats = _json.loads(_OURS_FALLBACK_PATH.read_text())
    OURS_MEAN_GRAY = float(_stats["mean_gray"])
    OURS_STD_GRAY = float(_stats["std_gray"])
    OURS_STATS_SOURCE = f"FALLBACK {_OURS_FALLBACK_PATH}"
else:
    # Hard defaults to keep import side-effect-free if both JSONs missing
    OURS_MEAN_GRAY = 0.115
    OURS_STD_GRAY = 0.167
    OURS_STATS_SOURCE = "HARDCODED_FALLBACK (folder 03 EchoCare measured values)"
# [OURS PATCH 1CH END]

# CLAHE clip_limit MUST be a (lo, hi) tuple with lo == hi to pin the
# value. Passing a scalar to A.CLAHE in albumentations >=2.0 is silently
# interpreted as the range (1.0, scalar) and sampled per call — that
# breaks val determinism AND drifts from src/training/preprocessing.py
# (which uses cv2.createCLAHE(clipLimit=2.0), deterministic). Tuple of
# equal floats is the documented way to get a deterministic clip.
_CLAHE_CLIP_LIMIT = (2.0, 2.0)
_CLAHE_TILE = (8, 8)
_CLIP_LO_PCT = 1.0
_CLIP_HI_PCT = 99.0

VALID_PREPROCESS = (
    "imagenet_default",
    "clahe_clip",
    "clahe_clip_sharpen_speckle",
    # [USFM PATCH BEIT05 2026-06-03] — IDENTICAL surface to
    # clahe_clip_sharpen_speckle (CLAHE LAB-L + percentile clip 1-99 +
    # RandBC + GaussNoise + Sharpen + MultNoise) EXCEPT the final
    # Normalize uses BEiT 0.5/0.5/0.5 instead of ImageNet. The single
    # variable vs the leader recipe is the encoder-matched normalization
    # (folder-07 USFM run B). 3-channel output.
    "clahe_clip_sharpen_speckle_beit05",
    # [SSL PATCH UNIMATCH-NB00 2026-05-23] — UniMatch v2 weak/strong augs
    # for folder-02 NB00. Both are PHOTOMETRIC-ONLY (NO spatial ops) so
    # landmark heatmaps stay aligned across the two views.
    # ssl_weak_v1   = same surface as clahe_clip_sharpen_speckle (folder-01
    #                  leader recipe). Re-exported under SSL name for clarity.
    # ssl_strong_v1 = ssl_weak_v1 + stronger RandomBrightnessContrast +
    #                  stronger GaussNoise + GaussianBlur + CoarseDropout.
    #                  CoarseDropout removes rectangular regions (fills 0)
    #                  WITHOUT moving any pixel = safe for landmark
    #                  consistency.
    "ssl_weak_v1",
    "ssl_strong_v1",
    # [USFM PATCH SSL-BEIT05 2026-06-10] — BEiT-0.5 twins of the UniMatch
    # unlabeled legs. Byte-identical aug surface to ssl_weak_v1 / ssl_strong_v1;
    # ONLY the final Normalize is BEiT 0.5/0.5/0.5 instead of ImageNet. Required
    # so the USFM conf-only UniMatch run (folder-07 nb02) feeds its encoder the
    # SAME normalization on labeled + unlabeled legs (encoder-native 0.5).
    "ssl_weak_v1_beit05",
    "ssl_strong_v1_beit05",
    # [OURS PATCH 1CH 2026-05-24] — 1-channel greyscale variant of
    # `clahe_clip_sharpen_speckle` for folder 04 (UltraDINO faithful 1ch
    # encoder per PROGRESSION_PLAN_v3 §3.2 fix A'). Identical pre-normalize
    # surface to the 3ch supervised leader (CLAHE LAB-L + percentile clip +
    # brightness/contrast + gauss-noise + sharpen + speckle); diverges ONLY
    # at the tail: ToGray(num_output_channels=1) + Normalize(OURS_MEAN_GRAY,
    # OURS_STD_GRAY) → tensor shape (1, H, W). Used by nb00 R07-1ch baseline
    # and Phase B downstream LP-FT of nb01a/b.
    "clahe_clip_sharpen_speckle_1ch_ours",
    # [NO-CLAHE ABLATION 2026-05-24] — same tail as
    # `clahe_clip_sharpen_speckle` (RandBC + GaussNoise + Sharpen +
    # MultNoise + ImageNet Normalize) but SKIPS the CLAHE + percentile
    # clip head. Used by folder-01 nb07-v2 to test whether CLAHE was
    # net-helpful in the chain or a net drag carried since #2.
    "sharpen_speckle_no_clahe",
    # [OURS PATCH SSL-PREPROC-MATCH 2026-06-01] — 1ch + OURS-norm variant of
    # the above, i.e. IDENTICAL to `clahe_clip_sharpen_speckle_1ch_ours`
    # EXCEPT it SKIPS the CLAHE + percentile-clip head. Rationale: the
    # folder-04 SSL encoder (nb01a) was pretrained on RAW greyscale (no
    # CLAHE, no percentile-clip) normalized with OURS_MEAN/STD measured on
    # raw images. Feeding CLAHE'd input into the same OURS normalize is an
    # input-distribution mismatch. This key matches the encoder's pretrain
    # distribution: Resize → (train photometric: RandBC + GaussNoise +
    # Sharpen + MultNoise) → ToGray(1) → Normalize(OURS) → ToTensor.
    "sharpen_speckle_1ch_ours_noclahe",
    # [PREPROC ABLATION 2026-06-16] — folder-01 preprocessing study (ANALISIS/preproc_study).
    # All 3 share the BASIC supervised train-aug tail (RandBC + GaussNoise) + ImageNet Normalize,
    # so ONLY the deterministic preprocessing head differs vs `clahe_clip` (clean A/B). Doppler-safe:
    # despeckle/unsharp act on LAB-L only (us_preproc._on_L). US speckle is multiplicative ->
    # despeckle BEFORE contrast/sharpen (paper rule, see _readingv2/09_us_preprocessing).
    "clahe_unsharp",        # Resize -> CLAHE(L) -> pctclip -> unsharp(L)   [contrast + edge]
    "clahe_bilateral",      # Resize -> bilateral(L) -> CLAHE(L) -> pctclip [despeckle -> contrast]
    "nlm_clahe_unsharp",    # Resize -> NLM(L) -> CLAHE(L) -> pctclip -> unsharp(L) [despeckle->contrast->edge]
)


def _percentile_clip(image, **_kwargs):
    """Clamp pixel values to [p1, p99] of the per-image distribution.

    CLIP (clamp), NOT stretch — division by (hi - lo) is the landmine.
    Preserves dtype (works on uint8 or float). p_lo / p_hi pinned to the
    registry defaults (1.0, 99.0).
    """
    lo = float(np.percentile(image, _CLIP_LO_PCT))
    hi = float(np.percentile(image, _CLIP_HI_PCT))
    return np.clip(image, lo, hi).astype(image.dtype)


def build_transforms(
    preprocess: str = "imagenet_default",
    input_size: int = 518,
    training: bool = True,
) -> A.Compose:
    """Build the A.Compose pipeline for a given preprocess name.

    Args
    ----
    preprocess: one of VALID_PREPROCESS. Unknown → KeyError listing valid.
    input_size: target HxW for A.Resize (518 = DINOv2 ViT-S native).
    training: if True, includes photometric aug; if False, val pipeline.
    """
    if preprocess not in VALID_PREPROCESS:
        raise KeyError(
            f"unknown preprocess {preprocess!r}; "
            f"valid: {VALID_PREPROCESS}"
        )

    ops = [A.Resize(input_size, input_size)]

    # [SSL PATCH UNIMATCH-NB00 2026-05-23] — SSL aug pipelines.
    # ssl_weak_v1 is the canonical weak-aug for the UniMatch teacher branch
    # (folder-01 leader stack minus spatial ops). ssl_strong_v1 layers
    # stronger photometric perturbations + CoarseDropout for the student
    # branch. Both follow the same head structure as clahe_clip_sharpen_speckle
    # (CLAHE LAB-L + percentile clip 1-99), so the teacher gets the
    # already-validated supervised preprocess.
    if preprocess in ("clahe_clip", "clahe_clip_sharpen_speckle",
                      # [USFM PATCH BEIT05] — beit05 inherits the same CLAHE head
                      "clahe_clip_sharpen_speckle_beit05",
                      "ssl_weak_v1", "ssl_strong_v1",
                      # [USFM PATCH SSL-BEIT05] — beit05 unlabeled twins inherit CLAHE head
                      "ssl_weak_v1_beit05", "ssl_strong_v1_beit05",
                      # [OURS PATCH 1CH] — 1ch_ours inherits CLAHE head
                      "clahe_clip_sharpen_speckle_1ch_ours"):
        ops.append(
            A.CLAHE(
                clip_limit=_CLAHE_CLIP_LIMIT,
                tile_grid_size=_CLAHE_TILE,
                p=1.0,
            )
        )
        ops.append(A.Lambda(image=_percentile_clip, p=1.0, name="percentile_clip_1_99"))

    # [PREPROC ABLATION 2026-06-16] — deterministic Doppler-safe preprocessing heads.
    # Order enforces the paper rule: despeckle (NLM/bilateral) BEFORE contrast (CLAHE) BEFORE
    # sharpen (unsharp). All on LAB-L (us_preproc._on_L). Applied at train AND val (p=1.0).
    elif preprocess in ("clahe_unsharp", "clahe_bilateral", "nlm_clahe_unsharp"):
        from src.training import us_preproc as _usp
        _clahe = A.CLAHE(clip_limit=_CLAHE_CLIP_LIMIT, tile_grid_size=_CLAHE_TILE, p=1.0)
        _pct = A.Lambda(image=_percentile_clip, p=1.0, name="percentile_clip_1_99")
        if preprocess == "clahe_unsharp":
            ops += [_clahe, _pct, A.Lambda(image=_usp.unsharp_l, p=1.0, name="unsharp_l")]
        elif preprocess == "clahe_bilateral":
            ops += [A.Lambda(image=_usp.bilateral_l, p=1.0, name="bilateral_l"), _clahe, _pct]
        else:  # nlm_clahe_unsharp
            ops += [A.Lambda(image=_usp.nlm_l, p=1.0, name="nlm_l"), _clahe, _pct,
                    A.Lambda(image=_usp.unsharp_l, p=1.0, name="unsharp_l")]

    if training:
        if preprocess in ("ssl_weak_v1", "ssl_weak_v1_beit05"):
            # Weak-aug branch for the UniMatch TEACHER stream. Same surface
            # as folder-01 leader stack EXCEPT GaussNoise std_range is set
            # EXPLICITLY to (0.02, 0.04) instead of inheriting the alb 2.x
            # default (0.2, 0.44). With the default, weak noise would be ~4×
            # STRONGER than ssl_strong_v1's explicit (0.03, 0.08), inverting
            # the invariance gradient direction (student forced to match a
            # noisier teacher with a cleaner strong input). Fix per
            # adversarial code review 2026-05-23 21:30.
            # NB: This makes ssl_weak_v1 NO LONGER byte-equivalent to
            # `clahe_clip_sharpen_speckle` — the supervised labeled batch
            # SHOULD use `clahe_clip_sharpen_speckle` directly (Run 07
            # pinned), NOT ssl_weak_v1, when reproducing the leader trajectory.
            ops += [
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(std_range=(0.02, 0.04), p=0.1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1), per_channel=False,
                    elementwise=True, p=0.2,
                ),
            ]
        elif preprocess in ("ssl_strong_v1", "ssl_strong_v1_beit05"):
            # Strong-aug branch: ssl_weak_v1 base + stronger perturbations
            # to force the student to extract invariant features. Plus
            # CoarseDropout (removes rectangular regions, does NOT move
            # pixels → safe for landmark coordinate consistency).
            ops += [
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.8,
                ),
                A.GaussNoise(std_range=(0.03, 0.08), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.MultiplicativeNoise(
                    multiplier=(0.85, 1.15), per_channel=False,
                    elementwise=True, p=0.3,
                ),
                # [SSL PATCH UNIMATCH-NB00 fix 2026-05-23] — albumentations
                # 2.x deprecated `max_holes`/`max_height`/etc. New API:
                # `num_holes_range`/`hole_height_range`/`hole_width_range`/`fill`.
                # Same effect as old: 1-8 holes 8-32 px each, fill 0.
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0,
                    p=0.3,
                ),
            ]
        else:
            # Original supervised paths (imagenet_default, clahe_clip,
            # clahe_clip_sharpen_speckle) — UNCHANGED, byte-equivalent to
            # prior behaviour.
            ops += [
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
            ]
            # [OURS PATCH 1CH] — 1ch_ours shares train tail with
            # clahe_clip_sharpen_speckle (sharpen + speckle); only the
            # post-aug normalization tail differs (handled below).
            if preprocess in ("clahe_clip_sharpen_speckle",
                              # [USFM PATCH BEIT05] — same sharpen+speckle train tail
                              "clahe_clip_sharpen_speckle_beit05",
                              "clahe_clip_sharpen_speckle_1ch_ours",
                              "sharpen_speckle_no_clahe",
                              # [OURS PATCH SSL-PREPROC-MATCH] same train tail
                              "sharpen_speckle_1ch_ours_noclahe"):
                ops += [
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                    A.MultiplicativeNoise(
                        multiplier=(0.9, 1.1),
                        per_channel=False,
                        elementwise=True,
                        p=0.2,
                    ),
                ]

    # [OURS PATCH 1CH BEGIN] — tail: 1ch_ours diverges from 3ch ImageNet
    # tail. Collapse RGB → 1-channel greyscale, then Normalize with the
    # OURS-corpus scalar stats. Output tensor shape (1, H, W).
    if preprocess in ("clahe_clip_sharpen_speckle_1ch_ours",
                      # [OURS PATCH SSL-PREPROC-MATCH] same 1ch+OURS tail,
                      # only the CLAHE head (skipped above) differs.
                      "sharpen_speckle_1ch_ours_noclahe"):
        ops += [
            A.ToGray(num_output_channels=1, p=1.0),
            A.Normalize(mean=(OURS_MEAN_GRAY,), std=(OURS_STD_GRAY,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    elif preprocess in ("clahe_clip_sharpen_speckle_beit05",
                        "ssl_weak_v1_beit05", "ssl_strong_v1_beit05"):
        # [USFM PATCH BEIT05] — encoder-native 0.5/0.5/0.5 norm, 3-channel
        # (USFM patch_embed is 3ch; greyscale already replicated to 3ch by
        # the dataset). Identical to the ImageNet tail except mean/std.
        ops += [
            A.Normalize(mean=BEIT_MEAN, std=BEIT_STD),
            ToTensorV2(),
        ]
    else:
        ops += [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    # [OURS PATCH 1CH END]
    return A.Compose(ops)


# ======================================================================
# [AUG ABLATION nb07 2026-06-16] — KEYPOINT-AWARE labeled transform.
# ----------------------------------------------------------------------
# `build_kp_transforms` replicates the EXACT `clahe_clip_sharpen_speckle`
# TRAIN ops above (CLAHE LAB-L + percentile-clip head, RandomBrightness
# Contrast p0.2, GaussNoise p0.1, Sharpen p0.3, MultiplicativeNoise p0.2,
# Resize 518, ImageNet Normalize, ToTensor) but as an A.Compose with
# `keypoint_params` so a geometric aug (Affine) can move both the image
# AND the keypoints. With `aug_cfg` empty (or geo prob 0) the geometric
# leg is absent → the photometric/normalize surface is byte-identical to
# `build_transforms('clahe_clip_sharpen_speckle', training=...)`. Combined
# with the matching dataset (KpAwareKeypointDataset) that normalizes
# keypoints by `input_size` (NOT input_size-1), the geo-OFF label equals
# the plain baseline KeypointDataset label — so nb04 itself is the control.
#
# Why a separate builder (not extend build_transforms): build_transforms
# returns an image-only Compose. Adding keypoint_params there would force
# every caller to pass keypoints. This keeps the nb04 path untouched and
# isolates the ablation surface.
# ======================================================================

# aug_cfg keys -> meaning (all single-variable; one arm sets exactly one):
#   "geo":           {"p": float}  Affine rotate±15 scale0.9-1.1 translate±10 REFLECT_101
#   "geo_rotate_only": {"p": float} Affine rotate±15 only (scale/translate off)
#   "shadow_gain":   {"p": float}  custom multiplicative intensity FIELD (no pixel move)
#   "bright_contrast_strong": {"p": float} stronger RandomBrightnessContrast on top of nb04's light one
# --- progressive menu (2026-06-17) — same single-variable contract, default-off ---
#   KP-MOVING (handled by Compose keypoint_params, border CONSTANT to avoid the
#   albu-2.x keypoint-reflection bug; see Affine note below):
#     "elastic":         {"p": float} ElasticTransform (non-rigid anatomy warp)
#     "grid_distortion": {"p": float} GridDistortion (non-rigid grid warp)
#     "hflip":           {"p": float} HorizontalFlip — ⚠️ swaps L/R anatomy (cardiac risk)
#     "vflip":           {"p": float} VerticalFlip
#   IMAGE-ONLY (keypoint-safe, no pixel move of landmarks):
#     "gamma":           {"p": float} RandomGamma (non-linear brightness curve)
#     "blur":            {"p": float} GaussianBlur (machine post-proc / denoise sim)
#     "fan_mask":        {"p": float} sector/corner occlusion field (US fan boundary)
#     "cutout":          {"p": float} CoarseDropout-style random rectangles to 0
#     "aspect":          {"p": float, "lo": float, "hi": float} anisotropic X-only
#                        rescale about the centre (aspect / stretch jitter)
_KP_AUG_KEYS = (
    "geo", "geo_rotate_only", "shadow_gain", "bright_contrast_strong",
    "elastic", "grid_distortion", "hflip", "vflip",
    "gamma", "blur", "fan_mask", "cutout", "aspect",
)


def _shadow_gain_field(image, **_kwargs):
    """Keypoint-SAFE multiplicative intensity field (acoustic shadow + depth gain).

    Multiplies the image by a smooth spatial gain map in [~0.55, ~1.25]:
      - a depth-gain ramp (vertical OR lateral, smooth linear) emulating
        TGC / depth attenuation differences across US machines, and
      - a soft low-intensity wedge ("acoustic shadow") at a random lateral
        position, raised-cosine profile so it is MILD and does NOT always
        cover the landmark (random center + partial width).
    NO pixel is moved → keypoints are unchanged → keypoint-safe. Operates on
    the uint8 RGB array; returns same dtype. Deterministic ops only use the
    albumentations-managed global RNG via np.random (Lambda is applied under
    the Compose's seeded context only for geometric ops; for this intensity
    field we draw from np.random which the train loop seeds via set_seed)."""
    h, w = image.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    # --- depth/lateral gain ramp (mild): pick orientation + slope sign ---
    if np.random.rand() < 0.5:
        ramp = yy * np.ones((1, w), dtype=np.float32)      # vertical
    else:
        ramp = xx * np.ones((h, 1), dtype=np.float32)      # lateral
    slope = np.random.uniform(-0.22, 0.22)                 # mild
    gain = 1.0 + slope * (ramp - 0.5) * 2.0                # ~[0.78, 1.22]

    # --- soft acoustic-shadow wedge (raised cosine), random lateral pos ---
    cx = np.random.uniform(0.15, 0.85)                     # center, not always on landmark
    half = np.random.uniform(0.08, 0.20)                   # partial width
    depth = np.random.uniform(0.20, 0.45)                  # how dark (mild)
    dist = np.abs(xx - cx) / max(half, 1e-3)
    wedge = np.where(dist < 1.0, 0.5 * (1.0 + np.cos(np.pi * dist)), 0.0).astype(np.float32)
    shadow = 1.0 - depth * wedge                           # >= 1-depth (>= 0.55)

    field = (gain * shadow).astype(np.float32)             # [H, W]
    out = image.astype(np.float32) * field[..., None]
    return np.clip(out, 0, 255).astype(image.dtype)


def _fan_mask_field(image, **_kwargs):
    """Keypoint-SAFE sector/corner occlusion (US fan boundary). Zeros a random
    triangular wedge anchored at one image corner — mimics the dark out-of-fan
    region of a sector probe. NO pixel is moved → keypoints unchanged. May cover
    a landmark (intentional: occlusion robustness). Draws from np.random (seeded
    by the train loop's set_seed), same convention as _shadow_gain_field."""
    h, w = image.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :] * np.ones((h, 1), np.float32)
    # pick a corner; the wedge is the region beyond a random diagonal cut from it.
    corner = int(np.random.randint(0, 4))
    cut = np.random.uniform(0.18, 0.42)                    # how deep the bite is
    fx = xx if corner in (0, 2) else (1.0 - xx)
    fy = yy if corner in (0, 1) else (1.0 - yy)
    mask = ((fx + fy) < cut).astype(np.float32)            # triangle near the corner
    out = image.astype(np.float32) * (1.0 - mask)[..., None]
    return np.clip(out, 0, 255).astype(image.dtype)


def _cutout_field(image, **_kwargs):
    """Keypoint-SAFE CoarseDropout: 1-3 random rectangles set to 0. NO pixel
    move → keypoints unchanged (may overlap a landmark = occlusion robustness).
    Lambda (not A.CoarseDropout) to be albumentations-version-proof for the
    long autonomous run. Draws from np.random (seeded by set_seed)."""
    h, w = image.shape[:2]
    out = image.copy()
    n = int(np.random.randint(1, 4))
    for _ in range(n):
        rh = int(np.random.uniform(0.06, 0.18) * h)
        rw = int(np.random.uniform(0.06, 0.18) * w)
        y0 = int(np.random.randint(0, max(h - rh, 1)))
        x0 = int(np.random.randint(0, max(w - rw, 1)))
        out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


def build_kp_transforms(
    preprocess: str = "clahe_clip_sharpen_speckle",
    input_size: int = 518,
    training: bool = True,
    aug_cfg: dict | None = None,
) -> A.Compose:
    """Keypoint-aware Compose: nb04 `clahe_clip_sharpen_speckle` surface + ONE aug.

    Args
    ----
    preprocess: must be 'clahe_clip_sharpen_speckle' (the nb04 labeled recipe).
                Other keys raise — this builder ONLY mirrors the leader recipe.
    input_size: 518 (DINOv2 ViT-S native). Resize target.
    training:   True -> photometric aug + optional geo aug; False -> deterministic
                head only (CLAHE+clip+Resize+Normalize), geo OFF (== nb04 val).
    aug_cfg:    dict toggling exactly one new aug (see _KP_AUG_KEYS). Empty / None
                / {"geo": {"p": 0.0}} -> geo-OFF -> byte-equiv photometric surface
                to build_transforms('clahe_clip_sharpen_speckle', training).

    Returns an A.Compose with keypoint_params=KeypointParams('xy',
    remove_invisible=False). Feed keypoints in 0..input_size after the dataset
    has already scaled orig-px -> 518 (Resize is inside the Compose, so feed
    ORIGINAL-pixel keypoints; the Compose Resize scales them).
    """
    # [USFM BEIT05 KP-AUG] accept the beit05 twin. The head + aug body are
    # byte-identical to 'clahe_clip_sharpen_speckle'; ONLY the tail Normalize
    # differs (BEiT 0.5/0.5/0.5 vs ImageNet) — exactly the same relationship the
    # image-only build_transforms beit05 twin has (see lines 343-354). Required
    # because USFM's correct norm is beit05 ([-1,+1] range); ImageNet would feed
    # USFM [-2.1,+2.6] = wrong range = corrupted features.
    _ALLOWED_KP_PREPROCESS = ("clahe_clip_sharpen_speckle",
                              "clahe_clip_sharpen_speckle_beit05")
    if preprocess not in _ALLOWED_KP_PREPROCESS:
        raise KeyError(
            "build_kp_transforms only mirrors the clahe_clip_sharpen_speckle "
            f"recipe (ImageNet or beit05 twin); got {preprocess!r}. "
            f"Valid: {_ALLOWED_KP_PREPROCESS}."
        )
    aug_cfg = dict(aug_cfg or {})
    bad = [k for k in aug_cfg if k not in _KP_AUG_KEYS]
    if bad:
        raise KeyError(f"unknown aug_cfg keys {bad}; valid: {_KP_AUG_KEYS}")

    # --- deterministic head (train AND val): Resize -> CLAHE -> pct-clip ---
    # IDENTICAL to build_transforms head for clahe_clip_sharpen_speckle.
    ops = [
        A.Resize(input_size, input_size),
        A.CLAHE(clip_limit=_CLAHE_CLIP_LIMIT, tile_grid_size=_CLAHE_TILE, p=1.0),
        A.Lambda(image=_percentile_clip, p=1.0, name="percentile_clip_1_99"),
    ]

    if training:
        # nb04 light photometric tail (byte-identical p-values to build_transforms).
        ops += [
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.MultiplicativeNoise(
                multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.2,
            ),
        ]
        # --- the ONE new aug (single variable per arm) ---
        if "bright_contrast_strong" in aug_cfg:
            p = float(aug_cfg["bright_contrast_strong"].get("p", 0.6))
            ops.append(A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=p))
        if "shadow_gain" in aug_cfg:
            p = float(aug_cfg["shadow_gain"].get("p", 0.4))
            ops.append(A.Lambda(image=_shadow_gain_field, p=p, name="shadow_gain"))
        # --- progressive menu: image-only (keypoint-safe) augs ---
        if "gamma" in aug_cfg:
            p = float(aug_cfg["gamma"].get("p", 0.5))
            ops.append(A.RandomGamma(gamma_limit=(70, 150), p=p))
        if "blur" in aug_cfg:
            p = float(aug_cfg["blur"].get("p", 0.4))
            ops.append(A.GaussianBlur(blur_limit=(3, 7), p=p))
        if "fan_mask" in aug_cfg:
            p = float(aug_cfg["fan_mask"].get("p", 0.4))
            ops.append(A.Lambda(image=_fan_mask_field, p=p, name="fan_mask"))
        if "cutout" in aug_cfg:
            p = float(aug_cfg["cutout"].get("p", 0.4))
            ops.append(A.Lambda(image=_cutout_field, p=p, name="cutout"))
        # [ASPECT JITTER 2026-07-23] anisotropic X-only rescale about the centre.
        # WHY: every task is fed through A.Resize(518,518) from a NON-square frame
        # (aspect 1.33-1.62), and the de-split cardiac panels of enrich_v8 are
        # crop-then-stretched by a further ~1.9-2.1x in x. So the model sees the
        # SAME anatomy at two different horizontal scales depending on whether the
        # frame was split or flat. `geo`'s scale is ISOTROPIC and cannot cover that;
        # this key jitters x alone so aspect becomes a nuisance variable instead of
        # a fixed per-subpopulation bias. Keypoint-moving -> lives in the kp-aware
        # builder only, border CONSTANT (same albu-2.x reflection guard as Affine).
        # Default range [0.80, 1.25] is bounded by a LABEL-SAFETY measurement, not
        # taste: over the 22022 enrich_v8 train landmarks, the worst-case share
        # pushed outside the 518 frame is 0.40% at [0.80,1.25] vs 0.97% at
        # [0.75,1.35] — and that mass is concentrated in HC (5.41%), PSAX (2.98%)
        # and fetal_femur (2.95%), i.e. exactly the tasks we can least afford to
        # feed clipped targets. `remove_invisible=False` keeps such points, so an
        # out-of-frame landmark becomes an empty heatmap = silent label noise.
        if "aspect" in aug_cfg:
            import cv2 as _cv2
            _a = aug_cfg["aspect"]
            _lo, _hi = float(_a.get("lo", 0.80)), float(_a.get("hi", 1.25))
            ops.append(A.Affine(scale={"x": (_lo, _hi), "y": (1.0, 1.0)},
                                border_mode=_cv2.BORDER_CONSTANT,
                                p=float(_a.get("p", 0.5))))
        # --- progressive menu: KP-MOVING augs (border CONSTANT, after Resize) ---
        if "hflip" in aug_cfg:
            ops.append(A.HorizontalFlip(p=float(aug_cfg["hflip"].get("p", 0.5))))
        if "vflip" in aug_cfg:
            ops.append(A.VerticalFlip(p=float(aug_cfg["vflip"].get("p", 0.5))))
        if "elastic" in aug_cfg or "grid_distortion" in aug_cfg:
            import cv2 as _cv2  # CONSTANT border: same keypoint-reflection guard as Affine
            if "elastic" in aug_cfg:
                p = float(aug_cfg["elastic"].get("p", 0.5))
                ops.append(A.ElasticTransform(
                    alpha=120.0, sigma=12.0, border_mode=_cv2.BORDER_CONSTANT, p=p))
            if "grid_distortion" in aug_cfg:
                p = float(aug_cfg["grid_distortion"].get("p", 0.5))
                ops.append(A.GridDistortion(
                    num_steps=5, distort_limit=0.3, border_mode=_cv2.BORDER_CONSTANT, p=p))
        # geometric (must move keypoints) — placed AFTER Resize so it works in 518-space.
        _geo_p = None
        _rotate_only = False
        if "geo" in aug_cfg:
            _geo_p = float(aug_cfg["geo"].get("p", 0.5))
        elif "geo_rotate_only" in aug_cfg:
            _geo_p = float(aug_cfg["geo_rotate_only"].get("p", 0.7))
            _rotate_only = True
        if _geo_p is not None and _geo_p > 0.0:
            import cv2 as _cv2
            # border_mode=BORDER_CONSTANT (NOT REFLECT_101): in albumentations
            # 2.x, A.Affine with REFLECT_101 reflects the KEYPOINTS across the
            # tile borders too, multiplying 1 kp into a 3×3 mosaic (2 pts -> 18)
            # and corrupting the label. CONSTANT fills out-of-frame IMAGE pixels
            # with 0 and keeps the keypoint count/positions correct. Landmarks
            # are interior, so constant-fill at the edges is harmless. (This is
            # a deliberate deviation from the literal "REFLECT_101" spec, forced
            # by an albu-2.x keypoint-reflection bug — verified empirically.)
            _aff = dict(rotate=(-15, 15), border_mode=_cv2.BORDER_CONSTANT, p=_geo_p)
            if not _rotate_only:
                _aff.update(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1))
            ops.append(A.Affine(**_aff))

    # [USFM BEIT05 KP-AUG] tail Normalize is the ONLY thing that varies by key:
    # BEiT 0.5 for the beit05 twin (USFM-native [-1,+1]), ImageNet otherwise.
    # Everything above (head + photometric + geo aug body) is byte-identical.
    if preprocess == "clahe_clip_sharpen_speckle_beit05":
        _norm = A.Normalize(mean=BEIT_MEAN, std=BEIT_STD)
    else:
        _norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ops += [_norm, ToTensorV2()]
    return A.Compose(
        ops,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
# [AUG ABLATION nb07 2026-06-16 END]
