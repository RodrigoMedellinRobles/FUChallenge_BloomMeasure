"""nb01b_dapt_lpft_ema_seed42.py — vendored clone of folder-04
`nb00_run07_1ch_baseline.py` with ONE added `[OURS PATCH SSL INIT]`
block: an `encoder_init_from_ckpt` hook that loads the iBOT-DAPT'd SSL
backbone (`ultradino_ours_vits14_seed42.pth`, nb01a Phase A output) into
`model.encoder` right after construction.

ROLE
----
Folder 04 SSL downstream Phase B. **Byte-equivalent to nb00 EXCEPT the
encoder init source.** This is the SHIP candidate of the single-variable
A/B:

  AB pair = (nb00_run07_1ch.pth) vs (nb01b_dapt_lpft_ema_seed42.pth)
  Single variable = encoder init swap (LVD-142M web → iBOT-DAPT'd 187k 40ep)

[OURS PATCH SSL INIT] — the SSL ckpt is a backbone-only state_dict (175
keys, `teacher.backbone.` prefix already stripped by nb01a extract). It
maps to `model.encoder` keys by PREPENDING `backbone.` (DINOv2Backbone
wraps the timm vit under `.backbone`). Two key-level deltas handled:
  - `mask_token` (iBOT-only) is DROPPED (encoder has no such param).
  - `pos_embed` is interpolated 257->1370 tokens (SSL global crops=224 ->
    16x16+1; downstream runs 518 -> 37x37+1) via timm
    `resample_abs_pos_embed` (bicubic) — same scheme DINOv2 uses for
    variable-resolution inference. ALL encoder weights thus come from
    SSL (single-variable-clean A/B).
The load is asserted strict: every encoder param must be filled by an
SSL tensor (missing==[]), and the only tolerated unexpected key is
`mask_token`. Any other mismatch raises before training.

Per PROGRESSION_PLAN_v3 §3.2 fix A' (1ch end-to-end faithful UltraDINO):
NO 1ch->3ch bridge confound. SSL pretrain (nb01a) and downstream LP-FT
(nb00 + nb01b) all run with `in_chans=1`.

OURS PATCH BLOCKS LAYERED ON nb07
---------------------------------
  [OURS PATCH 1CH PREPROC]  preprocess default forced to
      `clahe_clip_sharpen_speckle_1ch_ours` (1-channel + scalar
      OURS_MEAN/STD tail). albu_transforms module-top auto-loads stats
      from `experiments/04_dinov2_ultradinoLIKE/ours_norm_stats.json`.
  [OURS PATCH 1CH FACTORY]  import switched from organizer
      `model_factory.MultiTaskModelFactory` to LOCAL vendored
      `model_factory_patched.MultiTaskModelFactory` (adds `in_chans` kwarg).
  [OURS PATCH 1CH CHANNELS]  MultiTaskModelFactory(...) call passes
      `in_chans=1`. timm.create_model auto-adapts DINOv2-S/14 LVD-142M
      patch_embed weights via adapt_input_conv()
      (`.sum(dim=1, keepdim=True)`, preserves magnitude).

ALL OTHER lines below the file header are BYTE-EQUIVALENT to nb07.
Inherits [M0 PATCH], [M0 PATCH PREPROC], [M0 PATCH LPFT], [M0 PATCH EMA]
blocks from the folder-01 lineage (see folder-01/nb07 + nb06 docstrings).

Two-stage LP-FT call (from the wrapper notebook):
  Stage 1 — main(freeze_encoder=True,  head_lr=1e-3, ema_decay=None,    epochs=5)
  Stage 2 — main(freeze_encoder=False, init_from_ckpt=stage1.pth,
                 encoder_lr=1e-5, head_lr=5e-5, ema_decay=0.999,         epochs=35)

============================================================================
Original nb07 docstring continues below for traceability.
============================================================================
nb07_train_m0_lpft_ema.py — vendored copy of nb06_train_m0_lpft.py
with ONE additional [M0 PATCH EMA] block that adds Exponential Moving
Average (decay=0.999 default) of the model weights.

  - kwarg `ema_decay: float | None = None`
      When None: behavior is byte-identical to nb06 (no EMA, back-compat).
      When set: a `timm.utils.ModelEmaV3` is created around `model`
      AFTER any `init_from_ckpt` load AND AFTER any encoder freeze. The
      EMA shadow is updated once per `optimizer.step()`. Validation
      runs on the EMA shadow (`ema.module`), and the saved checkpoint
      is the EMA shadow's state_dict — NOT the raw model.

WHY (Polyak averaging on top of LP-FT):
  Two effects compound:
    1. Reduces variance of the val-epoch metric → less noisy best-ckpt
       selection (Tarvainen & Valpola 2017, Mean-Teacher; transferable
       to single-network supervised).
    2. Acts as implicit ensembling over the recent training trajectory.
       Stage 2 of LP-FT keeps the encoder near a stable feature
       manifold (Kumar fix), so EMA averaging over the late-stage
       trajectory tends to land at a flatter, lower-error point.

INHERITED FROM nb06_train_m0_lpft.py (unchanged):
  - All `[M0 PATCH LPFT]` blocks (freeze_encoder, init_from_ckpt,
    encoder_lr, head_lr)
  - All `[M0 PATCH PREPROC]` block (albumentations registry)
  - All earlier `[M0 PATCH]` blocks (split dispatch, save-path override,
    return-dict, CLI args)

NOT included (deliberate, single-variable disciplina):
  - No LLRD (one knob at a time; ARCH-01 showed LLRD-on-random-heads hurts)
  - No drop sigmoid (Run 05 territory; isolate EMA vs Run 06 first)
  - No SWA / SWA-LR (separate Polyak-family move)

DESIGN DECISIONS (from the adversarial review):

1. EMA buffer device: ModelEmaV3 defaults `device=None` → buffers live
   on the same device as `model` (cuda). No extra CPU<>GPU sync.

2. Stage 1 → stage 2 EMA reset: this script trains ONE stage at a
   time. The notebook calls main() twice. On the stage-2 call,
   `init_from_ckpt` loads the stage-1 ckpt into `model`, and the EMA
   wrapper is RE-CREATED here from scratch around the just-loaded
   model — so the EMA shadow at stage-2 epoch 0 equals the stage-1
   final weights, NOT some interpolation. This matches the
   adversarial-recommended "start fresh" behavior.

3. BN buffers in EMA vs raw weights: `exclude_buffers=False` (default)
   means ModelEmaV3 ALSO averages `running_mean` / `running_var`
   alongside trainable params. Evaluating against the EMA shadow's
   own BN buffers keeps the feature path consistent (EMA weights see
   EMA buffers, never a mix).

4. Save trigger: the existing best-MRE save trigger now writes
   `ema.module.state_dict()` instead of `model.state_dict()`. This is
   intentional — we want the EMA shadow as the final deliverable, not
   the raw model that the EMA was averaged from.

Every line not inside a `# [M0 PATCH]` / `# [M0 PATCH PREPROC]` /
`# [M0 PATCH LPFT]` / `# [M0 PATCH EMA]` block is verbatim from
nb06_train_m0_lpft.py.
"""
import argparse
from collections import defaultdict
from pathlib import Path  # [M0 PATCH] needed for the clean-split CSV reads

# [M0 PATCH PREPROC BEGIN] — module-top sys.path bootstrap. Module-top
# imports run before any `if __name__ == "__main__"` block, so any
# fix-up has to happen here. Two cases:
#   - notebook: nb_bootstrap already prepended both paths → these inserts
#     are no-ops (idempotent dedup check).
#   - CLI (`python nb07_train_m0_lpft_ema.py ...`): nothing has set up
#     the path → these inserts make both `from dataset import ...`
#     (organizer baseline dir) and `from src.training.albu_transforms
#     import ...` (vendored shared src, used inside main()) resolve.
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

# [OURS PATCH 1CH FACTORY BEGIN] — prepend folder 04 dir so the LOCAL
# `model_factory_patched.py` import resolves explicitly by name (the
# organizer baseline `model_factory.py` lacks the in_chans kwarg).
_folder04 = Path(__file__).resolve().parent
if str(_folder04) not in _sys.path:
    _sys.path.insert(0, str(_folder04))
# [OURS PATCH 1CH FACTORY END]

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler
# [OURS PATCH 1CH FACTORY] — explicit local-vendored factory with in_chans kwarg
from model_factory_patched import MultiTaskModelFactory
from utils import evaluate_keypoint, keypoint_collate_fn, set_seed

# [OURS PATCH 1CH SAFETY 2026-05-24] — MEDIUM 6 adv fix: assert dataset.py
# resolves to organizer baseline, NOT a shadowed local copy. Folder 04 will
# host nb01a SSL data plumbing soon; a co-located `dataset.py` in folder 04
# would silently override KeypointDataset and break the AB vs nb07.
import dataset as _ds_module
assert "Foundation-Model-for-Ultrasound-Biometry/baseline" in _ds_module.__file__, (
    f"dataset.py shadowed by non-baseline path: {_ds_module.__file__}. "
    "Remove or rename the offending file before training."
)


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS =40
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
    # [OURS PATCH 1CH PREPROC] — default forced to 1ch+OURS scalar-norm
    # registry entry. Run 07 used "clahe_clip_sharpen_speckle" (3ch
    # ImageNet); folder 04 nb00 = same surface aug, 1ch + OURS tail.
    preprocess: str = "clahe_clip_sharpen_speckle_1ch_ours",
    # [M0 PATCH PREPROC END]
    # [M0 PATCH LPFT BEGIN]
    freeze_encoder: bool = False,
    init_from_ckpt: str | None = None,
    encoder_lr: float | None = None,
    head_lr: float | None = None,
    # [M0 PATCH LPFT END]
    # [OURS PATCH SSL INIT BEGIN] — load SSL-pretrained backbone into the
    # encoder (nb01b single-variable swap vs nb00). None = no-op =
    # byte-identical to nb00. Applied AFTER model construction and BEFORE
    # `init_from_ckpt` (stage-2 full-model load overrides it, which is
    # fine: stage-1 ckpt already carries the SSL encoder). See module
    # docstring for the key-remap + pos_embed interpolation contract.
    encoder_init_from_ckpt: str | None = None,
    # [OURS PATCH SSL INIT END]
    # [M0 PATCH EMA BEGIN] — Polyak averaging knob.
    # None = no EMA (back-compat with nb06). When set, wrap model in
    # timm.utils.ModelEmaV3(model, decay=ema_decay), update post-step,
    # eval/save against the EMA shadow.
    ema_decay: float | None = None,
    # [M0 PATCH EMA END]
):
    metric_column = "MRE (pixels)"
    metric_label = "MRE"

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

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

    train_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )
    train_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    val_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=val_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
    )
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

    # [OURS PATCH 1CH CHANNELS BEGIN] — encoder init in 1-channel mode.
    # timm.create_model(..., in_chans=1) auto-adapts the pretrained
    # LVD-142M patch_embed.proj.weight from shape (384, 3, 14, 14) to
    # (384, 1, 14, 14) via adapt_input_conv() = .sum(dim=1, keepdim=True)
    # (preserves filter magnitude). Heads (HeatmapHead) and forward path
    # are UNCHANGED — the only structural delta vs Run 07 is patch_embed.
    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
        in_chans=1,
    ).to(device)
    # [OURS PATCH 1CH CHANNELS END]

    # [OURS PATCH SSL INIT BEGIN] — load iBOT-DAPT'd SSL backbone into
    # model.encoder. Runs BEFORE init_from_ckpt so that in stage 1 the
    # frozen encoder == SSL weights (heads train on top); in stage 2 the
    # stage-1 ckpt (which already carries the SSL encoder) is loaded
    # afterward by init_from_ckpt and supersedes this — so the notebook
    # only needs to pass encoder_init_from_ckpt in stage 1.
    if encoder_init_from_ckpt is not None:
        from timm.layers import resample_abs_pos_embed
        _ssl = torch.load(encoder_init_from_ckpt, map_location=device, weights_only=True)
        if isinstance(_ssl, dict) and "model" in _ssl and isinstance(_ssl["model"], dict):
            _ssl = _ssl["model"]
        if isinstance(_ssl, dict) and "state_dict" in _ssl:
            _ssl = _ssl["state_dict"]
        _ssl = dict(_ssl)
        _ssl.pop("mask_token", None)  # iBOT-only param; encoder has none

        _enc_sd = model.encoder.state_dict()  # keys are `backbone.<...>`
        _remap = {}
        for _k, _v in _ssl.items():
            _ek = f"backbone.{_k}"
            if _ek not in _enc_sd:
                raise RuntimeError(
                    f"[SSL INIT] SSL key '{_k}' -> '{_ek}' not in encoder "
                    f"state_dict (unexpected backbone structure). Aborting."
                )
            if _v.shape != _enc_sd[_ek].shape:
                if _k == "pos_embed":
                    # SSL 257 (16x16+1 @224) -> downstream 1370 (37x37+1 @518)
                    _tgt_tokens = _enc_sd[_ek].shape[1]
                    _grid = int(round((_tgt_tokens - 1) ** 0.5))  # 37
                    assert _grid * _grid + 1 == _tgt_tokens, (
                        f"[SSL INIT] non-square pos_embed grid: {_tgt_tokens}"
                    )
                    _v = resample_abs_pos_embed(
                        _v, new_size=[_grid, _grid], num_prefix_tokens=1,
                        interpolation="bicubic", antialias=True,
                    )
                    print(
                        f"[SSL INIT] pos_embed interpolated {tuple(_ssl['pos_embed'].shape)} "
                        f"-> {tuple(_v.shape)} (bicubic, {_grid}x{_grid} grid)"
                    )
                else:
                    raise RuntimeError(
                        f"[SSL INIT] shape mismatch on '{_ek}': SSL {tuple(_v.shape)} "
                        f"vs encoder {tuple(_enc_sd[_ek].shape)} (only pos_embed "
                        f"interp is supported). Aborting."
                    )
            _remap[_ek] = _v

        _missing, _unexpected = model.encoder.load_state_dict(_remap, strict=False)
        if _missing:
            raise RuntimeError(
                f"[SSL INIT] {len(_missing)} encoder params NOT filled by SSL ckpt "
                f"(first 5: {list(_missing)[:5]}). A/B would be confounded. Aborting."
            )
        if _unexpected:
            raise RuntimeError(
                f"[SSL INIT] unexpected keys after remap: {list(_unexpected)}. Aborting."
            )
        print(
            f"[SSL INIT] loaded {len(_remap)} backbone tensors into encoder from "
            f"{encoder_init_from_ckpt} (missing=0, unexpected=0)"
        )
    # [OURS PATCH SSL INIT END]

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

    # [M0 PATCH EMA BEGIN] — create EMA wrapper AFTER init_from_ckpt load
    # and AFTER freeze_encoder so the EMA copy mirrors the same param
    # graph the optimizer sees. ema_model.module is the EMA shadow we
    # use for val + final save. update_after_step=0 → start averaging
    # from the very first optimizer step; use_warmup=False keeps a
    # constant decay (we explicitly chose 0.999, no schedule).
    ema_model = None
    if ema_decay is not None:
        from timm.utils import ModelEmaV3
        ema_model = ModelEmaV3(
            model,
            decay=float(ema_decay),
            update_after_step=0,
            use_warmup=False,
            device=device,
            exclude_buffers=False,  # average BN running stats too
        )
        print(f"[EMA] ModelEmaV3 active, decay={ema_decay} (shadow on {device})")
    else:
        print("[EMA] disabled (ema_decay=None) — back-compat with nb06")
    # [M0 PATCH EMA END]

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_score = float("inf")
    best_epoch = -1
    _save_path = str(model_save_path) if model_save_path else MODEL_SAVE_PATH
    print(f"Best-checkpoint metric: {metric_label} (lower is better)")
    print("\n" + "=" * 50 + "\n--- Start Keypoint Training ---")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for batch in loop:
            images = batch["image"].to(device)
            task_ids = batch["task_id"]
            batch_loss_values = []

            for current_task_id in sorted(set(task_ids)):
                task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]
                task_images = images[task_indices]
                task_heatmaps = torch.stack([batch["heatmap"][i] for i in task_indices], 0).to(device)

                pred_logits = model(task_images, task_id=current_task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                loss = F.mse_loss(pred_heatmaps, task_heatmaps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # [M0 PATCH EMA BEGIN] — update EMA shadow after every
                # optimizer.step(). ModelEmaV3.update reads the raw
                # model params and pushes them into the shadow with
                # decay = ema_decay. No-op if ema_model is None.
                if ema_model is not None:
                    ema_model.update(model)
                # [M0 PATCH EMA END]

                loss_value = float(loss.item())
                batch_loss_values.append(loss_value)
                epoch_train_losses[current_task_id].append(loss_value)

            mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
            loop.set_postfix(loss=mean_batch_loss, groups=len(set(task_ids)), lr=scheduler.get_last_lr()[0])

        print(f"\n--- Epoch {epoch + 1} Average Train Loss ---")
        for task_id in sorted(epoch_train_losses.keys()):
            avg_loss = float(np.mean(epoch_train_losses[task_id]))
            print(f"  - {task_id}: {avg_loss:.4f}")

        # [M0 PATCH EMA BEGIN] — evaluate against EMA shadow if present.
        # evaluate_keypoint puts the passed module into eval mode and
        # iterates val_loader; passing ema_model.module means BN buffers
        # used at inference are the averaged ones too (consistent path).
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
            # [M0 PATCH EMA BEGIN] — save EMA shadow, NOT the raw model.
            # When ema_model is None, this falls back to the raw model
            # (byte-identical to nb06 behavior).
            _save_state = (
                ema_model.module.state_dict() if ema_model is not None
                else model.state_dict()
            )
            torch.save(_save_state, _save_path)
            # [M0 PATCH EMA END]
            print(f"-> New best model saved! {metric_label} improved to: {best_val_score:.4f}\n")

        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {_save_path}")
    # [M0 PATCH EMA BEGIN] — extend the return dict with EMA info so the
    # notebook wrapper can record it in summary.json.
    return {
        "best_val_mre": best_val_score,
        "best_epoch": best_epoch,
        "epochs": NUM_EPOCHS,
        "ema_decay": ema_decay,
        "ema_active": ema_model is not None,
        # [OURS PATCH 1CH] — surface 1ch + OURS-norm facts to summary.json
        "in_chans": 1,
        "preprocess_key": preprocess,
        "encoder_init": ENCODER,
        # [OURS PATCH SSL INIT] — record SSL backbone provenance for the A/B log
        "encoder_init_from_ckpt": encoder_init_from_ckpt,
    }
    # [M0 PATCH EMA END]


if __name__ == "__main__":
    # [M0 PATCH PREPROC]
    from src.training.albu_transforms import VALID_PREPROCESS

    parser = argparse.ArgumentParser(description="Keypoint training script")
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help="Validation ratio per task (0~1), e.g. 0.2 means each task uses 20%% for validation.",
    )
    # [M0 PATCH BEGIN]
    parser.add_argument("--leaky", action="store_true", default=True)
    parser.add_argument("--clean", dest="leaky", action="store_false")
    parser.add_argument("--train-split-csv", type=str, default=None)
    parser.add_argument("--val-split-csv", type=str, default=None)
    parser.add_argument("--model-save-path", type=str, default=None)
    # [M0 PATCH END]
    # [M0 PATCH PREPROC BEGIN]
    parser.add_argument(
        # [OURS PATCH 1CH] — CLI default forced to 1ch+OURS registry entry
        "--preprocess", type=str, default="clahe_clip_sharpen_speckle_1ch_ours",
        help=f"Preprocess pipeline name. Valid: {VALID_PREPROCESS}.",
    )
    # [M0 PATCH PREPROC END]
    # [M0 PATCH EMA BEGIN]
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="EMA decay for ModelEmaV3 (e.g. 0.999). Omit/None disables EMA.",
    )
    # [M0 PATCH EMA END]
    # [OURS PATCH SSL INIT BEGIN]
    parser.add_argument(
        "--encoder-init-from-ckpt", type=str, default=None,
        help="Path to SSL backbone ckpt to load into model.encoder "
             "(nb01b A/B swap). Omit/None = byte-identical to nb00.",
    )
    # [OURS PATCH SSL INIT END]
    args = parser.parse_args()

    main(
        val_split=float(args.val_split),
        leaky=bool(args.leaky),
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        model_save_path=args.model_save_path,
        preprocess=str(args.preprocess),
        ema_decay=args.ema_decay,
        # [OURS PATCH SSL INIT]
        encoder_init_from_ckpt=args.encoder_init_from_ckpt,
    )
