"""nb_usfm_train.py — vendored copy of nb07_train_m0_lpft_ema.py
(folder-01 LP-FT + EMA leader recipe) with ONE additional
[USFM PATCH ENC] block that swaps the encoder from DINOv2 ViT-S
(518/patch-14, ImageNet-DINOv2 pretrain) to **USFM** — the Universal
Ultrasound Foundation Model (Jiao et al., Medical Image Analysis 2024,
arXiv 2401.00153), a BEiT-B/16 MAE-pretrained on a 3M-image ultrasound
corpus.

WHY (the domain axis):
  Run 07 = DINOv2-S 18.4804 px DARK leader. DINOv2 is pretrained on
  natural images (LVD-142M). USFM is pretrained IN-DOMAIN (ultrasound,
  3M images). This isolates the PRETRAIN-DOMAIN axis with the encoder
  dropped into the byte-identical folder-01 LP-FT+EMA recipe. SUPERVISED
  only — no UniMatch, no unlabeled, no SSL on our data. It measures the
  pure supervised base of an in-domain US foundation encoder.

ENCODER FORMAT (BEiT, not vanilla ViT — see src/usfm_encoder.py):
  - timm shell: beit_base_patch16_224 with use_abs_pos_emb=False,
    use_shared_rel_pos_bias=True, use_rel_pos_bias=False, global_pool=''.
  - INPUT 512 (NOT 518): BEiT is patch-16, 518 does not divide 16.
    512/16 = 32 → 32x32 token grid (vs DINOv2 37x37 at 518). Chosen for
    landmark feature density (closest patch-16 grid to the baseline).
  - rel_pos_bias table is interpolated 14x14(pretrain)→32x32 by timm's
    beit.checkpoint_filter_fn. global_pool='' keeps USFM's real final
    LayerNorm in self.norm (applied by forward_features) — see the
    fc_norm trap documented in usfm_encoder.py.
  - Load result: missing=0, unexpected={mask_token} (MAE token, dropped).
  - The organizer DINOv2Backbone.forward already handles BEiT's single
    CLS-prefix tensor output → NO forward patch needed.

NORMALIZATION (A/B — resolved from USFM's own repo 2026-06-03):
  USFM's official downstream code normalizes with IMAGENET stats
  (openmedlab/USFM usdsgen/data/datasets.py:28-43), so ImageNet is the
  PAPER-FAITHFUL norm — it also happens to be folder-01 byte-identical.
    - `clahe_clip_sharpen_speckle`        → ImageNet = PRIMARY (paper-faithful
      + folder-01 leader recipe).
    - `clahe_clip_sharpen_speckle_beit05` → 0.5/0.5/0.5 = timm-BEiT-default
      WRONG-NORM CONTROL (measures the cost of the wrong norm; NOT native).
  Identical CLAHE chain; only the final Normalize differs. Run both.

LOAD SITE: USFM weights are loaded into model.encoder.backbone
  immediately AFTER MultiTaskModelFactory instantiation and BEFORE any
  init_from_ckpt (LP-FT) load. ENCODER_WEIGHTS=None so the factory calls
  timm.create_model(pretrained=False) — the local USFM .pth is the only
  init source. For stage 2 (init_from_ckpt set) the USFM preload is
  SKIPPED (the stage-1 ckpt already carries USFM-derived encoder weights)
  but the BEiT-512 architecture is still swapped in first so the
  state_dict shapes match.

INHERITED FROM nb07_train_m0_lpft_ema.py (unchanged):
  - All [M0 PATCH EMA] blocks (ema_decay, ModelEmaV3, EMA shadow eval/save).
  - All [M0 PATCH LPFT] blocks (freeze_encoder, init_from_ckpt, lrs).
  - All [M0 PATCH PREPROC] block (albumentations registry).
  - All earlier [M0 PATCH] blocks (split dispatch, save-path, CLI args).

NOT included (deliberate, single-variable disciplina):
  - No LR/WD rescale for the BEiT encoder; nb07 recipe kept byte-identical
    (encoder@1e-5, head@5e-5 stage-2; head@1e-3 stage-1). If USFM
    underfits at this LR, that itself is the answer.
  - No UniMatch / no unlabeled / no SSL. Supervised base only.

Every line not inside a [M0 PATCH*] / [USFM PATCH ENC] block is verbatim
from nb07_train_m0_lpft_ema.py.
"""
import argparse
from collections import defaultdict
from pathlib import Path  # [M0 PATCH] needed for the clean-split CSV reads

# [M0 PATCH PREPROC BEGIN] — module-top sys.path bootstrap. Two cases:
#   - notebook: nb_bootstrap already prepended both paths → no-ops.
#   - CLI: nothing has set up the path → these make `from dataset import`
#     (organizer baseline) and `from src.training.albu_transforms import`
#     (vendored shared src) resolve.
# [USFM PATCH ENC] also adds experiments/07_usfm/src as a DIRECT module
# dir (NOT a package) so `import usfm_encoder` resolves without colliding
# with the `src.` package namespace owned by _shared_src.
import sys as _sys
_here = Path(__file__).resolve().parent
while not (_here / "pyproject.toml").exists():
    if _here == _here.parent:
        raise RuntimeError("repo root not found (need pyproject.toml)")
    _here = _here.parent
for _p in (
    str(_here / "experiments" / "_shared_src"),
    str(_here / "Foundation-Model-for-Ultrasound-Biometry" / "baseline"),
    str(_here / "experiments" / "07_usfm" / "src"),  # [USFM PATCH ENC]
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

# [USFM PATCH ENC] — shared faithful USFM loader (factory + rel_pos interp).
from usfm_encoder import attach_usfm_encoder, default_usfm_weights_path


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 40
DATA_ROOT_PATH = "data"
# [USFM PATCH ENC BEGIN] — BEiT-B/16 shell for USFM. ENCODER_WEIGHTS=None
# → factory builds timm BEiT (pretrained=False, default img_size 224)
# only to size the heads at 768; attach_usfm_encoder then swaps in the
# img_size=512 BEiT and loads USFM weights. INPUT_SIZE=512 (patch-16).
ENCODER = "beit_base_patch16_224"
ENCODER_WEIGHTS = None
RANDOM_SEED = 42
MODEL_SAVE_PATH = "best_model.pth"
VAL_SPLIT = 0.2
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
INPUT_SIZE = 512
# [USFM PATCH ENC END]
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
    preprocess: str = "imagenet_default",
    # [M0 PATCH PREPROC END]
    # [M0 PATCH LPFT BEGIN]
    freeze_encoder: bool = False,
    init_from_ckpt: str | None = None,
    encoder_lr: float | None = None,
    head_lr: float | None = None,
    # [M0 PATCH LPFT END]
    # [M0 PATCH EMA BEGIN]
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

    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
    ).to(device)

    # [USFM PATCH ENC BEGIN] — swap the throwaway BEiT@224 backbone for the
    # img_size=512 USFM encoder. load_weights only in stage 1 (init_from_ckpt
    # is None); in stage 2 the init_from_ckpt full-model load supplies the
    # encoder weights, so we only need the matching BEiT-512 architecture.
    _usfm_report = attach_usfm_encoder(
        model,
        img_size=INPUT_SIZE,
        weights_path=default_usfm_weights_path(),
        device=device,
        load_weights=(init_from_ckpt is None),
    )
    model = model.to(device)
    print(f"[USFM PATCH ENC] {_usfm_report}")
    # [USFM PATCH ENC END]

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
    # and AFTER freeze_encoder so the EMA copy mirrors the same param graph.
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

                # [M0 PATCH EMA BEGIN] — update EMA shadow after every step.
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
            _save_state = (
                ema_model.module.state_dict() if ema_model is not None
                else model.state_dict()
            )
            torch.save(_save_state, _save_path)
            # [M0 PATCH EMA END]
            print(f"-> New best model saved! {metric_label} improved to: {best_val_score:.4f}\n")

        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {_save_path}")
    # [M0 PATCH EMA BEGIN] — extend return dict with EMA info.
    return {
        "best_val_mre": best_val_score,
        "best_epoch": best_epoch,
        "epochs": NUM_EPOCHS,
        "ema_decay": ema_decay,
        "ema_active": ema_model is not None,
        "encoder": "USFM_beit_base_patch16",
        "input_size": INPUT_SIZE,
    }
    # [M0 PATCH EMA END]


if __name__ == "__main__":
    # [M0 PATCH PREPROC]
    from src.training.albu_transforms import VALID_PREPROCESS

    parser = argparse.ArgumentParser(description="USFM keypoint training script")
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
        help=f"Preprocess pipeline name. Valid: {VALID_PREPROCESS}.",
    )
    # [M0 PATCH PREPROC END]
    # [M0 PATCH LPFT BEGIN]
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--init-from-ckpt", type=str, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    # [M0 PATCH LPFT END]
    # [M0 PATCH EMA BEGIN]
    parser.add_argument("--ema-decay", type=float, default=None)
    # [M0 PATCH EMA END]
    args = parser.parse_args()

    main(
        val_split=float(args.val_split),
        leaky=bool(args.leaky),
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
        model_save_path=args.model_save_path,
        preprocess=str(args.preprocess),
        freeze_encoder=bool(args.freeze_encoder),
        init_from_ckpt=args.init_from_ckpt,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        ema_decay=args.ema_decay,
    )
