"""nb04_train_unimatch_conf_only.py — fork of nb00_train_unimatch_minimum.py
adding ONE bundled technique on top of the NB00 minimum:

  (1) Per-task confidence-threshold pseudo-labeling — at the start of
      stage 2 we scan the val labeled split with the EMA teacher to
      compute per-task peak-height quantiles. During cons loss we mask
      out (sample, keypoint) pairs where the teacher's peak is below
      this τ_task. This kills the garbage-gradient pathway from
      low-confidence teacher predictions corrupting the student.

CUMULATIVE RECIPE (NB00 → NB04):
- NB00: weak+1strong consistency + EMA teacher + λ ramp + 1-step/batch refactor
  = 18.33 DARK MRE (current leader).
- NB04 ADDS: per-task confidence mask on the SINGLE strong-aug cons stream.
  NO Feature Perturbation (that was NB01, FAILED bundled +0.74 px).
  NO dual-strong, NO TPS (those were NB02, FAILED +1.65 px).

WHY NB04 (rationale 2026-05-25):
NB01 bundled FP+Conf and FAILED. NB02 bundled dual-strong+TPS and FAILED.
We don't know which knob in NB01 broke it (FP or Conf, or interaction).
NB04 isolates ONLY the confidence threshold to attribute the FAILURE in NB01:
- If NB04 ≤ NB00 (18.33 DARK) → Confidence is safe; FP was the culprit in NB01
- If NB04 > NB00 → Confidence is the culprit; mark Confidence FAILED too

ONE knob. Low overfit risk on val. Expected ROI: -0.1 to -0.3 px.

A/B target: NB00 DARK 18.33 px. Gate ≤18.1 strong / 18.1-18.4 marginal / >18.4 wash.

DELIBERATE DEVIATIONS FROM CANONICAL UniMatch (inherited from NB00)
-------------------------------------------------------------------
- Mean-Teacher EMA(0.999) teacher (NOT student.eval() self-distil).
- Single strong-aug branch (s1 only, no dual + no CutMix).
- MSE-on-sigmoid cons loss (NOT softmax-CE).

NEW KWARGS (NB04 only)
----------------------
- `enable_confidence: bool = True`      (per-task peak-height confidence mask)
- `conf_quantile: float = 0.25`         (low quantile of val-set teacher peaks → τ_task)
- `conf_max_samples_per_task: int = 256` (calibration cap)

INHERITED FROM NB00 (UNCHANGED):
- All `[SSL PATCH UNIMATCH-NB00]` blocks (unlabeled loader, ramp, cons loop)
- All `[M0 PATCH EMA]` blocks (ModelEmaV3 ctor + per-step update + save EMA shadow)
- All `[M0 PATCH LPFT]` blocks (freeze_encoder, init_from_ckpt, encoder_lr, head_lr)
- All `[M0 PATCH PREPROC]` blocks (albumentations registry)
- 1-step/outer-batch grad-accumulation refactor

τ DERIVATION CHOICE (documented per ssl_methodology_expert NB01 review F2):
τ_task = 25th-quantile of EMA teacher peaks on val LABELED images. This
overlaps with ckpt-selection images → τ may be slightly over-confident.
Defensible for NB04 (same as NB01); fix in future NB by switching to
held-out 5% unlabeled slice if confidence pathway turns out load-bearing.
"""
import argparse
from collections import defaultdict
from pathlib import Path

# [M0 PATCH PREPROC BEGIN] — module-top sys.path bootstrap
import sys as _sys
_here = Path(__file__).resolve().parent
while not (_here / "pyproject.toml").exists():
    if _here == _here.parent:
        raise RuntimeError("repo root not found (need pyproject.toml)")
    _here = _here.parent
for _p in (
    str(_here / "experiments" / "_shared_src"),
    str(_here / "Foundation-Model-for-Ultrasound-Biometry" / "baseline"),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# [M0 PATCH PREPROC END]

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler
from model_factory import MultiTaskModelFactory
from utils import evaluate_keypoint, keypoint_collate_fn, set_seed


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 40
DATA_ROOT_PATH = "data"
ENCODER = "vit_small_patch14_dinov2.lvd142m"
ENCODER_WEIGHTS = "pretrained"
RANDOM_SEED = 42
MODEL_SAVE_PATH = "best_model.pth"
VAL_SPLIT = 0.2
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
INPUT_SIZE = 518
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}

# [SSL PATCH UNIMATCH-NB00 BEGIN] — subdomain → task_id mapping for the
# unlabeled corpus. A4C1 + A4C2 are two ingestion paths of the same
# clinical task (raw 800×600 vs 256² preprocessed; see the package documentation
# §"Non-obvious things"). Everything else is identity. fetal_femur has no
# unlabeled subdomain — no SSL coverage there. Cons loss covers 8/9 tasks.
_SUBDOMAIN_TO_TASK = {
    "A4C1": "A4C",
    "A4C2": "A4C",
    "AOP":  "AOP",
    "FA":   "FA",
    "FUGC": "FUGC",
    "HC":   "HC",
    "IVC":  "IVC",
    "PLAX": "PLAX",
    "PSAX": "PSAX",
}
# [SSL PATCH UNIMATCH-NB00 END]


def _build_task_configs(dataframe):
    configs = []
    seen = set()
    for _, row in dataframe.iterrows():
        task_name = str(row["task_name"])
        task_id = str(row["task_id"])
        if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append(
            {
                "task_id": task_id,
                "task_name": "Regression",
                "num_classes": int(row["num_classes"]),
            }
        )
    if not configs:
        raise ValueError("No keypoint tasks found in dataset.")
    return configs


def _stratified_split_indices(dataframe, val_split: float, seed: int):
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError("val_split must be in (0, 1).")

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for _, group in dataframe.groupby("task_id", sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)

        total = len(indices)
        val_count = int(round(total * float(val_split)))
        if total >= 2:
            val_count = max(1, min(total - 1, val_count))
        else:
            val_count = 0

        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


# [M0 PATCH BEGIN] — clean split source
def _clean_split_indices(dataframe, train_csv: str, val_csv: str):
    import pandas as pd
    path_to_idx = {str(p): i for i, p in enumerate(dataframe["image_path"].astype(str))}
    train_paths = pd.read_csv(train_csv)["image_path"].astype(str).tolist()
    val_paths = pd.read_csv(val_csv)["image_path"].astype(str).tolist()
    missing = [p for p in train_paths + val_paths if p not in path_to_idx]
    if missing:
        raise KeyError(
            f"{len(missing)} split image_paths not in dataset (first 3: {missing[:3]})"
        )
    return [path_to_idx[p] for p in train_paths], [path_to_idx[p] for p in val_paths]
# [M0 PATCH END]


def main(
    val_split: float = VAL_SPLIT,
    # [M0 PATCH BEGIN]
    leaky: bool = True,
    train_split_csv: str | None = None,
    val_split_csv: str | None = None,
    model_save_path: str | Path | None = None,
    # [M0 PATCH END]
    # [M0 PATCH PREPROC BEGIN]
    preprocess: str = "imagenet_default",
    # [M0 PATCH PREPROC END]
    # [AUG ABLATION nb07 REBUILD 2026-06-16] — keypoint-aware aug on the LABELED
    # leg whose CONTROL is the EXISTING nb04 checkpoint (NO control re-run).
    # Default geo_loader=False → byte-identical to nb04 (KeypointDataset + albu
    # `preprocess`=clahe_clip_sharpen_speckle). When geo_loader=True, the LABELED
    # train+val datasets become KpAwareKeypointDataset — the SAME baseline data
    # flow + nb04's EXACT clahe_clip_sharpen_speckle surface, made keypoint-aware,
    # PLUS exactly one new aug from `aug_cfg`. With aug_cfg empty / geo p=0, the
    # geo-OFF label is byte-equivalent to nb04's label → nb04 IS the control.
    # The UniMatch unlabeled weak/strong legs are UNCHANGED (photometric only).
    # `aug_cfg`: dict with exactly one of {geo, geo_rotate_only, shadow_gain,
    # bright_contrast_strong} -> {"p": float}. The legacy geo_aug_p/geo_preprocess/
    # geo_rotate_only knobs are kept ONLY for backward signature compat; when
    # aug_cfg is given it takes precedence and the labeled preprocess is forced
    # to clahe_clip_sharpen_speckle (nb04 recipe).
    geo_loader: bool = False,
    aug_cfg: dict | None = None,
    geo_aug_p: float = 0.7,
    geo_preprocess: str = "clahe_clip_sharpen_speckle",
    geo_rotate_only: bool = False,
    # [AUG ABLATION nb07 END]
    # [M0 PATCH LPFT BEGIN]
    freeze_encoder: bool = False,
    init_from_ckpt: str | None = None,
    encoder_lr: float | None = None,
    head_lr: float | None = None,
    # [M0 PATCH LPFT END]
    # [M0 PATCH EMA BEGIN]
    ema_decay: float | None = None,
    # [M0 PATCH EMA END]
    # [SSL PATCH UNIMATCH-NB00 BEGIN] — UniMatch v2 minimum kwargs.
    enable_unimatch: bool = False,
    lambda_cons_max: float = 0.5,
    lambda_ramp_epochs: int = 7,
    unlabeled_batch_size: int = 4,
    num_workers_unlabeled: int = 4,
    unlabeled_data_root: str | None = None,
    unlabeled_weak_preprocess: str = "ssl_weak_v1",
    unlabeled_strong_preprocess: str = "ssl_strong_v1",
    # [SSL PATCH UNIMATCH-NB00 END]
    # [SSL PATCH UNIMATCH-NB04 BEGIN] — per-task Confidence-only.
    # NB04 = NB00 + per-task peak-height confidence mask. NO Feature
    # Perturbation (that was the FP half of NB01, kept OUT here to
    # isolate the Confidence knob).
    enable_confidence: bool = True,
    conf_quantile: float = 0.25,
    conf_max_samples_per_task: int = 256,
    # [SSL PATCH UNIMATCH-NB04 END]
):
    metric_column = "MRE (pixels)"
    metric_label = "MRE"

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # [SSL PATCH UNIMATCH-NB00 BEGIN] — hard precondition: cons loss needs
    # a teacher, and the teacher IS the EMA shadow (no duplicate EMA). If
    # the caller asks for UniMatch without EMA, RAISE immediately.
    if enable_unimatch and ema_decay is None:
        raise ValueError(
            "enable_unimatch=True requires ema_decay set (teacher = EMA shadow). "
            "Got ema_decay=None. Either pass ema_decay=0.999 (stage 2 path) "
            "or pass enable_unimatch=False (stage 1 LP path)."
        )
    # [SSL PATCH UNIMATCH-NB00 END]

    # [M0 PATCH PREPROC BEGIN]
    from src.training.albu_transforms import build_transforms, VALID_PREPROCESS
    print(f"Preprocess pipeline: {preprocess}  (valid: {VALID_PREPROCESS})")
    train_transforms = build_transforms(
        preprocess=preprocess, input_size=INPUT_SIZE, training=True
    )
    val_transforms = build_transforms(
        preprocess=preprocess, input_size=INPUT_SIZE, training=False
    )
    # [M0 PATCH PREPROC END]

    temp_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )

    task_configs = _build_task_configs(temp_dataset.dataframe)
    task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in task_configs}

    # [M0 PATCH BEGIN]
    if leaky:
        train_indices, val_indices = _stratified_split_indices(
            temp_dataset.dataframe,
            val_split=val_split,
            seed=RANDOM_SEED,
        )
    else:
        if not (train_split_csv and val_split_csv):
            raise ValueError(
                "leaky=False requires train_split_csv + val_split_csv "
                "(point them at experiments/SPLITS/{train,internal_val}.csv)"
            )
        train_indices, val_indices = _clean_split_indices(
            temp_dataset.dataframe, train_split_csv, val_split_csv
        )
    # [M0 PATCH END]
    train_size = len(train_indices)
    val_size = len(val_indices)

    # [AUG ABLATION nb07 REBUILD] geo_loader → keypoint-aware aug on labeled leg,
    # control = nb04 itself (KpAwareKeypointDataset; geo-OFF == nb04 label).
    if geo_loader:
        from src.training.dataset import KpAwareKeypointDataset
        # aug_cfg drives the single new aug. If a caller still passes the legacy
        # geo_aug_p path (aug_cfg is None), translate it into an aug_cfg so the
        # behavior is the keypoint-aware nb04-recipe loader (NOT USKeypointDataset).
        _aug_cfg = dict(aug_cfg) if aug_cfg else {}
        if not _aug_cfg and geo_aug_p and geo_aug_p > 0.0:
            _aug_cfg = ({"geo_rotate_only": {"p": float(geo_aug_p)}}
                        if geo_rotate_only else {"geo": {"p": float(geo_aug_p)}})
        print(f"[AUG ABLATION REBUILD] geo_loader=True  aug_cfg={_aug_cfg}  "
              f"labeled preprocess=clahe_clip_sharpen_speckle (nb04 recipe)  (labeled leg only)")
        train_dataset = KpAwareKeypointDataset(
            data_root=DATA_ROOT_PATH, heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA,
            input_size=INPUT_SIZE, training=True, aug_cfg=_aug_cfg,
            preprocess="clahe_clip_sharpen_speckle",
        )
        val_dataset = KpAwareKeypointDataset(
            data_root=DATA_ROOT_PATH, heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA,
            input_size=INPUT_SIZE, training=False, aug_cfg={},
            preprocess="clahe_clip_sharpen_speckle",
        )
    else:
        train_dataset = KeypointDataset(
            data_root=DATA_ROOT_PATH,
            transforms=train_transforms,
            heatmap_size=HEATMAP_SIZE,
            sigma=HEATMAP_SIGMA,
        )
        val_dataset = KeypointDataset(
            data_root=DATA_ROOT_PATH,
            transforms=val_transforms,
            heatmap_size=HEATMAP_SIZE,
            sigma=HEATMAP_SIGMA,
        )
    train_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)
    val_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    print(
        f"Dataset split (per-task stratified, val_split={val_split:.3f}): "
        f"{train_size} training samples, {val_size} validation samples"
    )

    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)

    train_sampler = KeypointUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    # [SSL PATCH UNIMATCH-NB00 BEGIN] — build unlabeled weak/strong loader.
    unlabeled_loader = None
    unlabeled_iter = None
    if enable_unimatch:
        from src.training.nb02_00_unimatch_dataset import (
            UnlabeledWeakStrongDataset,
            make_unlabeled_loader,
        )
        _unl_data_root = (
            Path(unlabeled_data_root) if unlabeled_data_root else Path(DATA_ROOT_PATH)
        )
        weak_tf = build_transforms(
            preprocess=unlabeled_weak_preprocess, input_size=INPUT_SIZE, training=True,
        )
        strong_tf = build_transforms(
            preprocess=unlabeled_strong_preprocess, input_size=INPUT_SIZE, training=True,
        )
        unl_dataset = UnlabeledWeakStrongDataset(
            data_root=_unl_data_root,
            weak_transform=weak_tf,
            strong_transform=strong_tf,
        )
        unlabeled_loader = make_unlabeled_loader(
            dataset=unl_dataset,
            batch_size=unlabeled_batch_size,
            num_workers=num_workers_unlabeled,
            seed=RANDOM_SEED,
            persistent_workers=False,
        )
        unlabeled_iter = iter(unlabeled_loader)
        print(
            f"[UniMatch] unlabeled corpus: {len(unl_dataset):,} images, "
            f"batch={unlabeled_batch_size}, weak={unlabeled_weak_preprocess}, "
            f"strong={unlabeled_strong_preprocess}, "
            f"λ_max={lambda_cons_max}, ramp_epochs={lambda_ramp_epochs}"
        )
    else:
        print("[UniMatch] disabled (enable_unimatch=False) — byte-equiv to nb07")
    # [SSL PATCH UNIMATCH-NB00 END]

    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
    ).to(device)

    # [M0 PATCH LPFT BEGIN] — load init checkpoint (stage 2 of LP-FT)
    if init_from_ckpt is not None:
        _init_state = torch.load(init_from_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(_init_state)
        print(f"[LPFT] loaded init weights from {init_from_ckpt}")
    # [M0 PATCH LPFT END]

    # [M0 PATCH LPFT BEGIN] — param-group builder
    _enc_lr = encoder_lr if encoder_lr is not None else LEARNING_RATE * 0.2
    _head_lr = head_lr if head_lr is not None else LEARNING_RATE * 10.0
    if freeze_encoder:
        for _p in model.encoder.parameters():
            _p.requires_grad_(False)
        n_enc_frozen = sum(1 for _p in model.encoder.parameters() if not _p.requires_grad)
        n_enc_total = sum(1 for _p in model.encoder.parameters())
        print(f"[LPFT] encoder FROZEN ({n_enc_frozen}/{n_enc_total} params requires_grad=False)")
        param_groups = []
        for task_id, head in model.heads.items():
            param_groups.append({"params": head.parameters(), "lr": _head_lr})
            print(f"[LPFT] Task head {task_id} LR: {_head_lr}")
    else:
        for _p in model.encoder.parameters():
            _p.requires_grad_(True)
        param_groups = [{"params": model.encoder.parameters(), "lr": _enc_lr}]
        for task_id, head in model.heads.items():
            param_groups.append({"params": head.parameters(), "lr": _head_lr})
            print(f"[LPFT] Task head {task_id} LR: {_head_lr}, encoder LR: {_enc_lr}")
    # [M0 PATCH LPFT END]

    # [M0 PATCH EMA BEGIN] — create EMA wrapper AFTER init_from_ckpt + freeze
    ema_model = None
    if ema_decay is not None:
        from timm.utils import ModelEmaV3
        ema_model = ModelEmaV3(
            model,
            decay=float(ema_decay),
            update_after_step=0,
            use_warmup=False,
            device=device,
            exclude_buffers=False,
        )
        print(f"[EMA] ModelEmaV3 active, decay={ema_decay} (shadow on {device})")
    else:
        print("[EMA] disabled (ema_decay=None) — back-compat with nb06")
    # [M0 PATCH EMA END]

    # [SSL PATCH UNIMATCH-NB00 BEGIN] — ramp setup.
    steps_per_epoch = len(train_loader)
    ramp_steps_total = int(lambda_ramp_epochs * steps_per_epoch)
    outer_step = 0
    if enable_unimatch:
        print(
            f"[UniMatch] λ ramp: 0 → {lambda_cons_max} over {ramp_steps_total} "
            f"outer steps ({lambda_ramp_epochs} ep × {steps_per_epoch} steps/ep)"
        )
    # [SSL PATCH UNIMATCH-NB00 END]

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_score = float("inf")
    best_epoch = -1
    _save_path = str(model_save_path) if model_save_path else MODEL_SAVE_PATH
    print(f"Best-checkpoint metric: {metric_label} (lower is better)")
    print("\n" + "=" * 50 + "\n--- Start Keypoint Training ---")

    # [SSL PATCH UNIMATCH-NB00 BEGIN refactor 2026-05-23 EVE] — telemetry
    epoch_cons_losses = []
    sup_optim_steps = 0
    cons_backwards_total = 0
    cons_tasks_per_batch = []
    # [SSL PATCH UNIMATCH-NB00 END]

    # [SSL PATCH UNIMATCH-NB04 BEGIN] — per-task confidence threshold
    # calibration (look-but-don't-touch on val labeled split). Computes
    # τ_task = low-quantile of EMA-teacher peak heights ACROSS the val
    # labeled images. Done ONCE before the epoch loop. Skipped when
    # enable_confidence=False OR enable_unimatch=False OR ema_model is None.
    # IDENTICAL to NB01 conf calibration block — knob isolated for A/B.
    confidence_thresholds: dict[str, float] = {}
    if enable_unimatch and enable_confidence and ema_model is not None:
        all_task_ids = list(_SUBDOMAIN_TO_TASK.values())
        # Deduplicate while preserving order
        seen = set()
        all_task_ids = [t for t in all_task_ids if not (t in seen or seen.add(t))]
        # Also include any task the LABEL set covers but unlabeled doesn't
        # (e.g. fetal_femur) — gives a τ floor for safety.
        for t in sorted(set(task_id_to_name.keys())):
            if t not in all_task_ids:
                all_task_ids.append(t)
        print(
            f"[UniMatch NB04] calibrating per-task confidence thresholds "
            f"on val labeled (quantile={conf_quantile}, max_samples_per_task="
            f"{conf_max_samples_per_task})..."
        )
        confidence_thresholds = compute_per_task_thresholds(
            teacher_module=ema_model.module,
            val_loader=val_loader,
            device=device,
            task_ids=all_task_ids,
            quantile=conf_quantile,
            max_samples_per_task=conf_max_samples_per_task,
        )
        print("[UniMatch NB04] τ_task = " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(confidence_thresholds.items())
        ))
    elif enable_unimatch:
        print(
            f"[UniMatch NB04] confidence gating OFF "
            f"(enable_confidence={enable_confidence}, ema_active={ema_model is not None})"
        )
    conf_mask_keep_ratio_running: list[float] = []
    # [SSL PATCH UNIMATCH-NB04 END]

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        # [SSL PATCH UNIMATCH-NB00 BEGIN]
        epoch_cons_running = []
        # [SSL PATCH UNIMATCH-NB00 END]

        for batch in loop:
            images = batch["image"].to(device)
            task_ids = batch["task_id"]

            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — gradient-accumulation:
            # ONE optimizer.zero_grad / outer batch.
            optimizer.zero_grad()
            # [SSL PATCH UNIMATCH-NB00 END]

            # --- SUP forward(s) + backward(s) [grad-accumulating] ---
            batch_loss_values = []
            for current_task_id in sorted(set(task_ids)):
                task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]
                task_images = images[task_indices]
                task_heatmaps = torch.stack([batch["heatmap"][i] for i in task_indices], 0).to(device)

                pred_logits = model(task_images, task_id=current_task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                sup_loss = F.mse_loss(pred_heatmaps, task_heatmaps)
                sup_loss.backward()

                loss_value = float(sup_loss.item())
                batch_loss_values.append(loss_value)
                epoch_train_losses[current_task_id].append(loss_value)

            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — cons forward(s) +
            # backward(s) [grad-accumulating, λ-weighted].
            cons_batch_value = 0.0
            lam = 0.0
            n_cons_tasks_actual = 0
            if enable_unimatch:
                try:
                    unl_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unl_batch = next(unlabeled_iter)

                unl_subdomains = list(unl_batch["subdomain"])
                unl_task_ids = [
                    _SUBDOMAIN_TO_TASK.get(s, None) for s in unl_subdomains
                ]
                lam = lambda_ramp_linear(
                    current_step=outer_step,
                    ramp_steps=ramp_steps_total,
                    lambda_max=float(lambda_cons_max),
                )

                if lam > 0.0:
                    unique_cons_tasks = sorted(
                        set(t for t in unl_task_ids if t is not None)
                    )
                    n_cons_tasks_actual = len(unique_cons_tasks)
                    if n_cons_tasks_actual > 0:
                        weak_all = unl_batch["image_weak"].to(device, non_blocking=True)
                        strong_all = unl_batch["image_strong"].to(device, non_blocking=True)
                        # [SSL PATCH UNIMATCH-NB04] — n_streams=1 (single
                        # strong cons stream, NO FP second stream). cons_weight
                        # matches NB00 exactly; only the confidence_mask kwarg
                        # added below differs from NB00. Total cons gradient
                        # magnitude per outer batch is invariant to NB00 when
                        # mask is full-ones (degenerate τ=0 path).
                        cons_weight = lam / n_cons_tasks_actual
                        cons_task_values = []

                        for cons_task_id in unique_cons_tasks:
                            sel = [i for i, t in enumerate(unl_task_ids) if t == cons_task_id]
                            weak = weak_all[sel]
                            strong = strong_all[sel]

                            # Teacher forward (EMA shadow, no grad).
                            with torch.no_grad():
                                teacher_logits = ema_model.module(weak, task_id=cons_task_id)
                                # [SSL PATCH UNIMATCH-NB04 BEGIN] — confidence
                                # mask from teacher peak heights. (B, K) in
                                # {0, 1} after threshold. When
                                # enable_confidence=False we pass None to fall
                                # back to NB00 behavior exactly.
                                if enable_confidence and confidence_thresholds:
                                    teacher_p_for_mask = torch.sigmoid(teacher_logits)
                                    conf_mask = per_task_confidence_mask(
                                        teacher_heatmap_sigmoid=teacher_p_for_mask,
                                        task_id=cons_task_id,
                                        thresholds=confidence_thresholds,
                                    )
                                    conf_mask_keep_ratio_running.append(
                                        float(conf_mask.mean().item())
                                    )
                                else:
                                    conf_mask = None
                                # [SSL PATCH UNIMATCH-NB04 END]
                            # Detach teacher graph for cons stream.
                            teacher_logits_d = teacher_logits.detach()

                            # --- SINGLE STREAM: strong-aug student (NB00 path)
                            #     with confidence mask added (NB04 patch).
                            student_logits = model(strong, task_id=cons_task_id)
                            cons_loss = unimatch_consistency_loss(
                                student_strong_logits=student_logits,
                                teacher_weak_logits=teacher_logits_d,
                                # [SSL PATCH UNIMATCH-NB04]
                                confidence_mask=conf_mask,
                            )
                            (cons_weight * cons_loss).backward()
                            cons_task_values.append(float(cons_loss.item()))
                            cons_backwards_total += 1

                        if cons_task_values:
                            cons_batch_value = float(np.mean(cons_task_values))
                            epoch_cons_running.append(cons_batch_value)

                cons_tasks_per_batch.append(n_cons_tasks_actual)
            # [SSL PATCH UNIMATCH-NB00 END]

            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — ONE optimizer.step +
            # ONE EMA update per outer batch.
            optimizer.step()
            sup_optim_steps += 1
            if ema_model is not None:
                ema_model.update(model)
            # [SSL PATCH UNIMATCH-NB00 END]

            mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
            if enable_unimatch:
                loop.set_postfix(
                    sup=mean_batch_loss,
                    cons=cons_batch_value,
                    lam=lam,
                    n_cons=n_cons_tasks_actual,
                    # [SSL PATCH UNIMATCH-NB04] — keep-ratio telemetry
                    keep=(conf_mask_keep_ratio_running[-1]
                          if (enable_confidence and conf_mask_keep_ratio_running) else 1.0),
                    lr=scheduler.get_last_lr()[0],
                )
            else:
                loop.set_postfix(
                    loss=mean_batch_loss,
                    groups=len(set(task_ids)),
                    lr=scheduler.get_last_lr()[0],
                )

            # [SSL PATCH UNIMATCH-NB00 BEGIN]
            outer_step += 1
            # [SSL PATCH UNIMATCH-NB00 END]

        print(f"\n--- Epoch {epoch + 1} Average Train Loss ---")
        for task_id in sorted(epoch_train_losses.keys()):
            avg_loss = float(np.mean(epoch_train_losses[task_id]))
            print(f"  - {task_id}: {avg_loss:.4f}")

        # [SSL PATCH UNIMATCH-NB00 BEGIN] — log mean cons loss this epoch
        if enable_unimatch and epoch_cons_running:
            epoch_cons_mean = float(np.mean(epoch_cons_running))
            epoch_cons_losses.append(epoch_cons_mean)
            print(
                f"  [UniMatch] cons_loss mean: {epoch_cons_mean:.5f}  "
                f"(λ_end={lam:.4f}, n_batches_with_cons={len(epoch_cons_running)})"
            )
        # [SSL PATCH UNIMATCH-NB00 END]

        # [M0 PATCH EMA BEGIN] — evaluate against EMA shadow if present
        _eval_model = ema_model.module if ema_model is not None else model
        val_results_df = evaluate_keypoint(_eval_model, val_loader, device, task_id_to_name)
        # [M0 PATCH EMA END]
        selected_val_score = float("inf")
        if not val_results_df.empty and metric_column in val_results_df.columns:
            selected_val_score = float(val_results_df[metric_column].mean())

        print(f"\n--- Epoch {epoch + 1} Validation Report ---")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val {metric_label} (Lower is better): {selected_val_score:.4f} ---")

        if selected_val_score < best_val_score:
            best_val_score = selected_val_score
            best_epoch = epoch
            # [M0 PATCH EMA BEGIN] — save EMA shadow, NOT the raw model
            _save_state = (
                ema_model.module.state_dict() if ema_model is not None
                else model.state_dict()
            )
            torch.save(_save_state, _save_path)
            # [M0 PATCH EMA END]
            print(f"-> New best model saved! {metric_label} improved to: {best_val_score:.4f}\n")

        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {_save_path}")
    # [SSL PATCH UNIMATCH-NB00 BEGIN] — extend return dict
    return {
        "best_val_mre": best_val_score,
        "best_epoch": best_epoch,
        "epochs": NUM_EPOCHS,
        "ema_decay": ema_decay,
        "ema_active": ema_model is not None,
        "unimatch_enabled": bool(enable_unimatch),
        "lambda_cons_max": float(lambda_cons_max) if enable_unimatch else None,
        "lambda_ramp_epochs": int(lambda_ramp_epochs) if enable_unimatch else None,
        "unlabeled_batch_size": int(unlabeled_batch_size) if enable_unimatch else None,
        "cons_loss_per_epoch": epoch_cons_losses if enable_unimatch else None,
        "outer_steps_total": outer_step,
        "sup_optim_steps_total": sup_optim_steps,
        "cons_backwards_total": cons_backwards_total,
        "cons_tasks_per_batch_mean": (
            float(np.mean(cons_tasks_per_batch)) if cons_tasks_per_batch else 0.0
        ),
        # [SSL PATCH UNIMATCH-NB04 telemetry] — Confidence-only knob.
        "confidence_enabled": bool(enable_confidence) if enable_unimatch else None,
        "confidence_quantile": float(conf_quantile) if (enable_unimatch and enable_confidence) else None,
        "confidence_thresholds": (
            {k: float(v) for k, v in confidence_thresholds.items()}
            if (enable_unimatch and enable_confidence and confidence_thresholds) else None
        ),
        "confidence_keep_ratio_mean": (
            float(np.mean(conf_mask_keep_ratio_running))
            if conf_mask_keep_ratio_running else None
        ),
    }
    # [SSL PATCH UNIMATCH-NB00 END]


# [SSL PATCH UNIMATCH-NB00 BEGIN] — re-export the cons-loss helper at
# module scope so the train loop above can use it without re-importing
# inside the hot path.
from src.training.nb02_00_unimatch_loss import (
    unimatch_consistency_loss,
    lambda_ramp_linear,
)
# [SSL PATCH UNIMATCH-NB00 END]
# [SSL PATCH UNIMATCH-NB04 BEGIN] — per-task confidence helpers (Conf only;
# NO Feature Perturbation helper imported — that was NB01's FP-failed path).
from src.training.nb02_01_per_task_confidence import (
    compute_per_task_thresholds,
    per_task_confidence_mask,
)
# [SSL PATCH UNIMATCH-NB04 END]


if __name__ == "__main__":
    # [M0 PATCH PREPROC]
    from src.training.albu_transforms import VALID_PREPROCESS

    parser = argparse.ArgumentParser(description="Keypoint training script + UniMatch v2 NB04 (Confidence-only)")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    # [M0 PATCH BEGIN]
    parser.add_argument("--leaky", action="store_true", default=True)
    parser.add_argument("--clean", dest="leaky", action="store_false")
    parser.add_argument("--train-split-csv", type=str, default=None)
    parser.add_argument("--val-split-csv", type=str, default=None)
    parser.add_argument("--model-save-path", type=str, default=None)
    # [M0 PATCH END]
    # [M0 PATCH PREPROC BEGIN]
    parser.add_argument(
        "--preprocess", type=str, default="imagenet_default",
        help=f"Valid: {VALID_PREPROCESS}.",
    )
    # [M0 PATCH PREPROC END]
    # [M0 PATCH EMA BEGIN]
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="EMA decay for ModelEmaV3 (e.g. 0.999). None disables EMA.",
    )
    # [M0 PATCH EMA END]
    # [SSL PATCH UNIMATCH-NB00 BEGIN]
    parser.add_argument("--enable-unimatch", action="store_true", default=False)
    parser.add_argument("--lambda-cons-max", type=float, default=0.5)
    parser.add_argument("--lambda-ramp-epochs", type=int, default=7)
    parser.add_argument("--unlabeled-batch-size", type=int, default=4)
    parser.add_argument("--num-workers-unlabeled", type=int, default=4)
    parser.add_argument("--unlabeled-data-root", type=str, default=None)
    parser.add_argument("--unlabeled-weak-preprocess", type=str, default="ssl_weak_v1")
    parser.add_argument("--unlabeled-strong-preprocess", type=str, default="ssl_strong_v1")
    # [SSL PATCH UNIMATCH-NB00 END]
    # [SSL PATCH UNIMATCH-NB04 CLI]
    parser.add_argument("--enable-confidence", action="store_true", default=True)
    parser.add_argument("--disable-confidence", dest="enable_confidence", action="store_false")
    parser.add_argument("--conf-quantile", type=float, default=0.25)
    parser.add_argument("--conf-max-samples-per-task", type=int, default=256)
    args = parser.parse_args()

    main(
        val_split=float(args.val_split),
        leaky=bool(args.leaky),
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        model_save_path=args.model_save_path,
        preprocess=str(args.preprocess),
        ema_decay=args.ema_decay,
        # [SSL PATCH UNIMATCH-NB00]
        enable_unimatch=bool(args.enable_unimatch),
        lambda_cons_max=float(args.lambda_cons_max),
        lambda_ramp_epochs=int(args.lambda_ramp_epochs),
        unlabeled_batch_size=int(args.unlabeled_batch_size),
        num_workers_unlabeled=int(args.num_workers_unlabeled),
        unlabeled_data_root=args.unlabeled_data_root,
        unlabeled_weak_preprocess=str(args.unlabeled_weak_preprocess),
        unlabeled_strong_preprocess=str(args.unlabeled_strong_preprocess),
        # [SSL PATCH UNIMATCH-NB04]
        enable_confidence=bool(args.enable_confidence),
        conf_quantile=float(args.conf_quantile),
        conf_max_samples_per_task=int(args.conf_max_samples_per_task),
    )
