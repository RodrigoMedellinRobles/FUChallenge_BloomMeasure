"""model_factory_patched.py — VENDORED LOCAL copy of organizer's
`Foundation-Model-for-Ultrasound-Biometry/baseline/model_factory.py`
with ONE single additional patch: `in_chans` kwarg on `DINOv2Backbone` +
`MultiTaskModelFactory` constructors.

Vendoring discipline (per `md_files/CLAUDE.md`):
  - NEVER modify upstream `Foundation-Model.../baseline/model_factory.py`.
  - Patches are LOCAL to this folder; marked with `# [OURS PATCH 1CH]`.
  - All non-patched code is BYTE-EQUIVALENT to upstream — every line
    outside `[OURS PATCH 1CH BEGIN]` / `[OURS PATCH 1CH END]` blocks is
    verbatim from `Foundation-Model.../baseline/model_factory.py`.

What changed
------------
  - `DINOv2Backbone.__init__` now accepts `in_chans: int = 3`.
    Default 3 = back-compat (byte-equivalent forward pass to upstream).
    When `in_chans != 3`, passed to `timm.create_model(...)`. timm
    internally calls `adapt_input_conv()` on the pretrained weights:
      * 3 → 1: `weight.sum(dim=1, keepdim=True)` (preserves magnitude)
      * 3 → N (N>3): broadcast + normalize
    Upstream patch_embed weights are NOT touched by hand; timm handles
    the adapt cleanly.
  - `MultiTaskModelFactory.__init__` now accepts `in_chans: int = 3` and
    forwards it to `DINOv2Backbone(...)`.

Used by
-------
  - `nb00_run07_1ch_baseline.{py,ipynb}` (in_chans=1, faithful Run 07
    recipe but 1-channel encoder + OURS_MEAN/STD norm).
  - Future `nb01a_dapt_pretrain_seed42.{py,ipynb}` Phase A (iBOT-DAPT
    student/teacher both in_chans=1).
  - Future `nb01b_dapt_lpft_ema_seed42.{py,ipynb}` Phase B (downstream
    LP-FT, loads Phase A teacher.pth into encoder, in_chans=1).

Per PROGRESSION_PLAN_v3 §3.2 fix A' (1ch end-to-end faithful UltraDINO).
"""
from typing import Dict, List

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class HeatmapHead(nn.Module):
    """Light decoder that maps DINOv2 feature maps to keypoint heatmaps."""

    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden = max(in_channels // 2, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DINOv2Backbone(nn.Module):
    """Returns the last patch feature map as a 2D tensor [B, C, H, W]."""

    # [OURS PATCH 1CH BEGIN] — added `in_chans: int = 3` kwarg
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        in_chans: int = 3,
    ):
        # [OURS PATCH 1CH END]
        super().__init__()
        timm = importlib.import_module("timm")
        # [OURS PATCH 1CH BEGIN] — pass in_chans to timm.create_model;
        # timm internally calls adapt_input_conv() on the pretrained
        # patch_embed weights (3→1 = .sum(dim=1, keepdim=True), preserves
        # magnitude). Single-line addition; original signature was
        # timm.create_model(model_name, pretrained=pretrained, num_classes=0).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )
        if in_chans != 3:
            print(
                f"[OURS PATCH 1CH] DINOv2Backbone in_chans={in_chans} "
                f"(timm auto-adapts patch_embed.proj.weight via adapt_input_conv)"
            )
        # [OURS PATCH 1CH END]
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
        self.out_channels = int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        # [OURS PATCH REG4] strip ALL prefix tokens (cls + register tokens),
        # not just cls. ViT-S = 1 (cls only) → back-compat with nb00/nb01b;
        # ViT-B reg4 = 5 (1 cls + 4 reg) → 518/14 leaves 1369 = 37² patches.
        n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                all_tokens = feats["x_prenorm"]
                patch_tokens = all_tokens[:, n_prefix:, :]
            else:
                raise RuntimeError("Unsupported forward_features output from DINOv2 backbone.")
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, n_prefix:, :]
            else:
                raise RuntimeError("Unexpected tensor shape from forward_features.")
        else:
            raise RuntimeError("Unexpected feature type from DINOv2 backbone.")

        bsz, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square; input size may be incompatible.")

        feat_map = patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)
        return feat_map


class MultiTaskModelFactory(nn.Module):
    """Keypoint heatmap model using a shared DINOv2 encoder and task-specific heads."""

    # [OURS PATCH 1CH BEGIN] — added `in_chans: int = 3` kwarg
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        task_configs: List[Dict],
        heatmap_size=(64, 64),
        in_chans: int = 3,
    ):
        # [OURS PATCH 1CH END]
        super().__init__()

        self.heatmap_size = heatmap_size

        print(f"Initializing DINOv2 encoder: {encoder_name}")
        # [OURS PATCH 1CH BEGIN] — forward in_chans to DINOv2Backbone
        self.encoder = DINOv2Backbone(
            model_name=encoder_name,
            pretrained=(encoder_weights is not None),
            in_chans=in_chans,
        )
        # [OURS PATCH 1CH END]

        self.heads = nn.ModuleDict()
        print(f"Creating keypoint heads for {len(task_configs)} tasks...")

        for config in task_configs:
            task_id = config["task_id"]
            task_name = config["task_name"]
            if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            num_points = int(config["num_classes"])
            self.heads[task_id] = HeatmapHead(in_channels=self.encoder.out_channels, num_points=num_points)

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        features = self.encoder(x)
        return self.heads[task_id](features, out_size=self.heatmap_size)
