"""usfm_encoder.py — faithful USFM (Universal Ultrasound Foundation Model)
BEiT-B/16 encoder loader for the FU_Biometry landmark stack.

USFM (Jiao et al., Medical Image Analysis 2024, arXiv 2401.00153) is a
ViT-B/16 pretrained with MAE on a 3M-image ultrasound corpus. Its
checkpoint is **BEiT-style**, NOT a vanilla timm ViT:

  - patch_embed.proj.weight = (768, 3, 16, 16)  → 3-channel, patch 16
  - NO abs pos_embed; uses a SHARED relative_position_bias (BEiT) tied
    to the 14x14 pretrain grid (224/16)
  - gamma_1/gamma_2 (LayerScale), attn q_bias/v_bias (no k_bias),
    cls_token, mask_token (MAE artifact), final LayerNorm named `norm`.

Correct timm shell: `beit_base_patch16_224` with
  use_abs_pos_emb=False, use_shared_rel_pos_bias=True,
  use_rel_pos_bias=False.

THREE non-obvious correctness points (verified on CPU 2026-06-03):

  1. INPUT 512 → grid 32x32. The shared rel_pos_bias table is tied to
     the window size: 14x14 pretrain → (2*14-1)^2+3 = 732 rows; 32x32 →
     (2*32-1)^2+3 = 3972 rows. timm's `beit.checkpoint_filter_fn`
     interpolates the table 732→3972 geometrically (and drops the
     stale relative_position_index buffer, which the model regenerates).
     Do NOT load the raw table — shape mismatch / silent skip.

  2. FINAL LAYERNORM ROUTING (the silent trap). timm BEiT sets
     `use_fc_norm = (global_pool == 'avg')` and the default global_pool
     IS 'avg'. With use_fc_norm=True, `self.norm = Identity` and the
     real final LN moves to `self.fc_norm` — which `forward_features`
     does NOT apply (it lives in forward_head, after pooling). The
     organizer DINOv2Backbone path calls only `forward_features`, so
     under the default the patch tokens would be returned UN-normalized
     and USFM's pretrained `norm.weight/bias` would land in a dropped
     Identity slot (shows up as MISSING fc_norm / UNEXPECTED norm).
     FIX: build with `global_pool=''` → use_fc_norm=False → self.norm
     is the real LN, loads USFM's `norm`, and forward_features applies
     it. Result: MISSING=[], UNEXPECTED=['mask_token'] (MAE token,
     correctly unused).

  3. FORWARD COMPAT. BEiT forward_features returns a tensor
     [B, 1+N, C] (single CLS prefix). The organizer
     DINOv2Backbone.forward already handles the tensor/dim==3 branch
     (`feats[:, 1:, :]` → reshape to square grid), so NO forward patch
     is needed — num_prefix_tokens == 1, 1024 patch tokens → 32x32.

NORMALIZATION (resolved 2026-06-03 from USFM's own repo — the paper-faithful
norm is ImageNet, NOT 0.5):
  USFM's official downstream pipeline normalizes with ImageNet stats —
  openmedlab/USFM `usdsgen/data/datasets.py:28-43`
  (`T.Normalize(IMAGENET_DEFAULT_MEAN/STD)` train+val, after Resize+ToTensor;
  no CLAHE). So despite USFM being a BEiT (timm default 0.5), the AUTHORS
  feed it ImageNet-normalized input downstream. The two preprocess variants:
    - `clahe_clip_sharpen_speckle`        → ImageNet norm = PAPER-FAITHFUL
      and folder-01 leader byte-identical (the PRIMARY run).
    - `clahe_clip_sharpen_speckle_beit05` → 0.5/0.5/0.5 = timm-BEiT-default
      WRONG-NORM CONTROL (measures the cost of the wrong norm; NOT native).
  This module does NOT pick a normalization — the preprocess registry does.
  Keep them consistent (reviewer flag F1).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Timm shell + kwargs that reproduce the USFM BEiT architecture.
USFM_TIMM_NAME = "beit_base_patch16_224"
USFM_BEIT_KWARGS = dict(
    use_abs_pos_emb=False,
    use_shared_rel_pos_bias=True,
    use_rel_pos_bias=False,
    global_pool="",          # → use_fc_norm=False → self.norm is the real final LN
)
USFM_EMBED_DIM = 768
# After checkpoint_filter_fn, the only acceptable leftover is the MAE
# mask_token (the model has no slot for it). Anything else is a real
# load gap and must be loud.
USFM_ALLOWED_UNEXPECTED = {"mask_token"}


def _find_repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__)).resolve()
    while not (p / "pyproject.toml").exists():
        if p == p.parent:
            raise RuntimeError("repo root not found (need pyproject.toml)")
        p = p.parent
    return p


def default_usfm_weights_path() -> Path:
    """Canonical location of the USFM encoder checkpoint."""
    return _find_repo_root() / "pretrained_weights" / "USFM_encoder.pth"


def build_usfm_backbone(
    img_size: int = 512,
    weights_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> torch.nn.Module:
    """Create a timm BEiT-B/16 at `img_size` and (optionally) load USFM weights.

    When `weights_path` is None the backbone is returned with random init
    (used in LP-FT stage 2, where the full-model `init_from_ckpt` overwrites
    it with the stage-1 weights — the architecture just has to match first).

    When `weights_path` is set the USFM checkpoint is filtered through timm's
    `beit.checkpoint_filter_fn` (rel_pos_bias 732→table-for-img_size resize +
    stale index drop), loaded with strict=False, and the leftovers are
    hard-asserted: MISSING must be empty and UNEXPECTED ⊆ {mask_token}.
    """
    timm = importlib.import_module("timm")
    beit_mod = importlib.import_module("timm.models.beit")

    backbone = timm.create_model(
        USFM_TIMM_NAME,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        **USFM_BEIT_KWARGS,
    )
    if int(backbone.num_features) != USFM_EMBED_DIM:
        raise RuntimeError(
            f"[USFM] unexpected embed dim {backbone.num_features} (expected {USFM_EMBED_DIM})"
        )

    if weights_path is not None:
        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(
                f"[USFM] encoder weights not found at {wp}. "
                "Expected pretrained_weights/USFM_encoder.pth."
            )
        raw = torch.load(wp, map_location="cpu", weights_only=False)
        state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        if not isinstance(state, dict):
            raise RuntimeError(f"[USFM] checkpoint at {wp} is not a state_dict (got {type(state)})")
        n_file = len(state)

        # timm-native BEiT filter: resizes the shared rel_pos_bias table to
        # this model's window size and drops the stale index buffer.
        filtered = beit_mod.checkpoint_filter_fn(state, backbone)
        missing, unexpected = backbone.load_state_dict(filtered, strict=False)

        unexpected_bad = [k for k in unexpected if k not in USFM_ALLOWED_UNEXPECTED]
        if verbose:
            print(
                f"[USFM] loaded {wp.name} (img_size={img_size})\n"
                f"[USFM]   file_keys={n_file}  filtered={len(filtered)}  "
                f"missing={len(missing)}  unexpected={len(unexpected)} "
                f"(allowed-dropped: {sorted(set(unexpected) & USFM_ALLOWED_UNEXPECTED)})"
            )
        if missing:
            raise RuntimeError(
                f"[USFM] LETHAL: {len(missing)} backbone keys missing after load. "
                f"First: {list(missing)[:5]}. The load target is the BEiT backbone "
                "(HeatmapHead keys live elsewhere and must NEVER appear here) — any "
                "missing key is a real architecture/key mismatch."
            )
        if unexpected_bad:
            raise RuntimeError(
                f"[USFM] LETHAL: {len(unexpected_bad)} unexpected ckpt keys not in "
                f"{sorted(USFM_ALLOWED_UNEXPECTED)}: {unexpected_bad[:5]}. "
                "USFM checkpoint format may have drifted."
            )

    return backbone.to(device)


def attach_usfm_encoder(
    model: torch.nn.Module,
    img_size: int = 512,
    weights_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    load_weights: bool = True,
    verbose: bool = True,
) -> Dict:
    """Swap a freshly-built MultiTaskModelFactory's encoder backbone for USFM.

    The factory is constructed with `encoder_name='beit_base_patch16_224'`
    (default img_size 224) only to size the HeatmapHeads at in_channels=768;
    this replaces `model.encoder.backbone` with the correctly-configured
    img_size=512 BEiT and (optionally) loads USFM weights. Heads are
    untouched — they are fully convolutional and channel-only dependent, so
    the spatial-resolution change (14x14→32x32) is transparent to them.

    `load_weights=False` (stage 2) → architecture-only swap; the caller's
    `init_from_ckpt` full-model load supplies the encoder weights.

    Returns a small report dict for logging.
    """
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "backbone"):
        raise RuntimeError("[USFM] model has no .encoder.backbone to swap (unexpected factory).")

    new_backbone = build_usfm_backbone(
        img_size=img_size,
        weights_path=(weights_path if load_weights else None),
        device=device,
        verbose=verbose,
    )
    enc.backbone = new_backbone
    enc.out_channels = int(new_backbone.num_features)  # 768 (unchanged; explicit)

    report = {
        "encoder": "USFM_beit_base_patch16",
        "img_size": img_size,
        "grid": img_size // 16,
        "out_channels": enc.out_channels,
        "weights_loaded": bool(load_weights and weights_path is not None),
        "weights_path": str(weights_path) if (load_weights and weights_path is not None) else None,
    }
    if verbose:
        print(f"[USFM] encoder attached: {report}")
    return report
