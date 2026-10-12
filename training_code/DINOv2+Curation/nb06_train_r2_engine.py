"""nb06_train_r2_engine.py — STEP 06 round-2 RETRAIN of the res700 FLAT champion
(LB ~24.44) on the nb05 prediction-guided CURATION manifest.

WHAT THIS IS
------------
This retrains the EXACT res700-flat champion recipe (folder-20
`hc_res700_flat_engine.py`: nb04 conf-only UniMatch, 2-stage LP-FT + EMA(0.999)
+ per-task conf gate, COLD-START stock DINOv2, cardiac FLAT, HC labeled leg @700)
changing ONLY THREE things driven by the round-2 curation manifest
(`runs/05_manifest_v1_0724/`):

  (1) WEIGHTED + AUGMENTED LABELED SET. Supervised leg = GT-labeled images
      (target = GT coords, per-sample weight ∈{1.0,0.6,0.25} from the manifest
      `sample_weight`) PLUS the SAFE-unlabeled images (target = their pseudo
      landmarks from `pseudo_targets.npz`, per-sample weight = manifest
      sample_weight = pseudo_weight; λp=1.0 so the pseudo_weight IS the
      per-sample weight). Built as a round-2 labeled CSV in the KeypointDataset
      schema (image_path, point_i_xy…, height, width, task_name, num_classes,
      task_id) + a `sample_weight` column. Safe-pseudo HC rows flow through the
      HC@700 leg exactly like GT HC rows.
  (2) CONSISTENCY POOL = AMBIGUOUS ONLY. The UniMatch unlabeled/consistency
      corpus is restricted to the 13,819 ambiguous image_ids (the champion's 8
      unlabeled manifests are subset to those ids). Outliers (1,575) are excluded
      from everything. Safe images are NOT in the consistency pool (they are now
      supervised).
  (3) PER-SAMPLE SUPERVISED WEIGHT. Each sample's supervised heatmap MSE is
      multiplied by its manifest weight (per-sample MSE, then weight, then mean),
      so a strong-suspicious GT contributes 0.25×, a low-Q pseudo contributes its
      pseudo_weight×. λ_cons=0.5 for the ambiguous consistency leg is UNCHANGED.

EVERYTHING ELSE IS BYTE-IDENTICAL to `hc_res700_flat_engine.py`: encoder
vit_small_patch14_dinov2.lvd142m (stock cold-start), preprocess
clahe_clip_sharpen_speckle, DARK σ=1.8, heatmap (64,64), Stage-1 LP (heads-only,
frozen enc, head_lr 1e-3, 5ep, EMA off, UniMatch off), Stage-2 FT (unfreeze,
enc_lr 1e-5, head_lr 5e-5, 35ep, EMA 0.999, UniMatch conf-only λ0.5 ramp7, conf
gate q0.25), HC labeled leg @700 (dynamic pos_embed), other 8 tasks @518.

HOW THE 3 CHANGES ARE WIRED (no fork of the champion patches)
-------------------------------------------------------------
* P0 (DATA_ROOT) is repointed to a round-2 2-CSV mirror (`_dataroot_r2`) whose
  csv/ globs to EXACTLY {round2_train_labeled.csv, internal_val.csv}.
* P1,P2,P3,P4,P6 (HC@700 machinery: globals / smoke-read / highres loader +
  dynamic backbone / sup-skip / smoke-cap) are IMPORTED VERBATIM from
  `hc_res700_flat_engine.py` — zero drift.
* P5' replaces the champion HC@700 backward with a PER-SAMPLE-WEIGHTED HC loss
  (change 3, HC leg).
* P_SUP replaces the mixed-518 batch-mean sup loss with a PER-SAMPLE-WEIGHTED sup
  loss (change 3, the other 8 tasks).
* The labeled dataset class + collate are swapped to `WeightedKeypointDataset` /
  `weighted_keypoint_collate_fn` via NAMESPACE INJECTION (late binding on the
  module globals `KeypointDataset` / `keypoint_collate_fn`) — this threads the
  per-sample `sample_weight` from the CSV → batch → loss for BOTH the mixed-518
  loader and the HC@700 loader, with NO extra string patches.
* Change 1's target coords + weights live in round2_train_labeled.csv; change 2's
  ambiguous restriction is a subset of the unlabeled manifests under
  `_dataroot_r2/unlabeled_root/csv_unlabeled/`.

TRAIN / VAL SEPARATION (IMPORTANT — read the FLAG)
--------------------------------------------------
The manifest's 6,743 GT images = 5,386 in the champion TRAIN split + 1,357 in the
champion INTERNAL_VAL split. VAL stays the champion internal_val.csv (checkpoint
selection — leaky, but it is the CHAMPION'S OWN val). We therefore do NOT add the
1,357 GT-in-val images to the training supervised set — training on the val set
would be a NEW train-on-val leak on top of the champion's own leak. So:

    round2 TRAINING labeled rows = 5,386 GT-in-train + 4,822 safe-pseudo = 10,208
    (the manifest 'supervised' regime total is 11,565 = 6,743 gt + 4,822 pseudo;
     the 1,357 GT-in-val are used ONLY for val, exactly as the champion uses them.
     21 champion-train fetal_femur rows are not in the curation manifest at all —
     they fall outside the curated 26,959-image corpus and are dropped, which is
     part of the curation change, not a separate confound.)

WARM-START (FLAG — default matches champion; orchestrator can flip)
-------------------------------------------------------------------
Per doc step 14, round-2 is a FRESH LP-FT COLD-START from stock DINOv2 (NOT warm
from the champion ckpt), same as the champion. `--warm-start-ckpt PATH` is exposed
for the orchestrator to override; default None = stock cold-start.

Run:
  python3 .../nb06_train_r2_engine.py --preflight-only
  python3 .../nb06_train_r2_engine.py --smoke   # 1+1 epochs, capped
  python3 .../nb06_train_r2_engine.py           # full 5+35 (GATED)
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
REPO = _here
while not (REPO / "pyproject.toml").exists():
    if REPO == REPO.parent:
        raise RuntimeError("repo root not found (need pyproject.toml)")
    REPO = REPO.parent
os.chdir(REPO)

# champion engine dir (import its verbatim HC@700 patches) + shared src + baseline + nb04
CHAMP_DIR = REPO / "experiments" / "experiments_collaborator" / "20_hc_res700"
for _p in (
    str(CHAMP_DIR),
    str(REPO / "experiments" / "_shared_src"),
    str(REPO / "Foundation-Model-for-Ultrasound-Biometry" / "baseline"),
    str(REPO / "experiments" / "02_m0_dinov2_SEMISUPERVISED" / "_models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402  (kept for parity; loss is inline below)

from dataset import KeypointDataset  # noqa: E402
from utils import keypoint_collate_fn  # noqa: E402
import hc_res700_flat_engine as CH  # noqa: E402  (champion — verbatim HC@700 patches)

NB04_SRC = (REPO / "experiments" / "02_m0_dinov2_SEMISUPERVISED" / "_models" /
            "nb04_train_unimatch_conf_only.py")

# ---- champion FLAT source (GT coords + val + unlabeled manifests) ----
CHAMP_FLAT = CHAMP_DIR / "_dataroot_flat"
CHAMP_TRAIN_CSV = CHAMP_FLAT / "csv" / "train.csv"
CHAMP_VAL_CSV = CHAMP_FLAT / "csv" / "internal_val.csv"
CHAMP_UNL_CSV_DIR = CHAMP_FLAT / "unlabeled_root" / "csv_unlabeled"

# ---- nb05b rebalanced curation manifest (the round-2 driver) ----
# NOCAP FORK: manifest comes from nb05b (pseudo-budget + IVC bypass applied on top
# of nb05's verbatim v1 partition). Env overridable for smoke runs.
MANIFEST_DIR = Path(os.environ.get(
    "R2_MANIFEST_DIR", str(_here.parent / "runs" / "05b_rebalance_nocap_v1_0725")))
MANIFEST_CSV = MANIFEST_DIR / "manifest_round2.csv"
PSEUDO_NPZ = MANIFEST_DIR / "pseudo_targets.npz"
NB00_MANIFEST = Path(os.environ.get(
    "R2_NB00_DIR", str(_here.parent / "runs" / "00_infer_nocap_v1_0725"))) / "manifest.csv"

# ---- round-2 build outputs ----
# Env-overridable like R2_MANIFEST_DIR / R2_NB00_DIR above. It was hardcoded, which
# silently decoupled the output path from run_all.sh's $TAG: the build wrote a complete
# dataroot under the old tag, exited 0, and the chain aborted because the file it was
# told to look for never appeared. Pre-existing — the 2026-07-25 run died in nb00 and
# never got far enough to hit it.
OUT_ROOT = Path(os.environ.get(
    "R2_OUT_ROOT", str(_here.parent / "runs" / "06_train_nocap_v1_0725")))
DATAROOT_R2 = OUT_ROOT / "_dataroot_r2"
R2_TRAIN_CSV = DATAROOT_R2 / "csv" / "round2_train_labeled.csv"
R2_VAL_CSV = DATAROOT_R2 / "csv" / "internal_val.csv"
R2_UNL_ROOT = DATAROOT_R2 / "unlabeled_root"

PREPROCESS = "clahe_clip_sharpen_speckle"
HC_INPUT_SIZE = 700  # champion HC labeled leg resolution (unchanged variable)
LAMBDA_PSEUDO = 1.0  # λp — pseudo target weight multiplier; 1.0 => pseudo_weight IS the weight

# per-task keypoint counts (num_classes). Pseudo vec len must equal 2*num_classes.
NUM_CLASSES = {"A4C": 16, "AOP": 4, "FA": 4, "FUGC": 2, "HC": 4, "IVC": 2,
               "PLAX": 22, "PSAX": 4, "fetal_femur": 2}

# ---------------------------------------------------------------- integrity gate
# NOCAP FORK: the v1 engine hard-coded the round-1 (cap=3000) regime totals here.
# With cap=0 the manifest covers the FULL unlabeled corpus (191 170), so the totals
# are data-dependent. They are now DERIVED from the manifest that nb05b wrote and
# cross-checked against nb05b's own sidecar counts, so the gate still fails loudly
# on a truncated/corrupt manifest -- it just no longer pins round-1 magic numbers.
EXP_SUP = EXP_SUP_GT = EXP_SUP_PSEUDO = EXP_CONS = EXP_IGNORED = EXP_TOTAL = None
EXP_GT_IN_TRAIN = EXP_R2_TRAIN_LABELED = None


def _derive_expected(man):
    """Populate the EXP_* gate from the manifest + nb05b sidecar (fail-loud)."""
    global EXP_SUP, EXP_SUP_GT, EXP_SUP_PSEUDO, EXP_CONS, EXP_IGNORED, EXP_TOTAL
    global EXP_GT_IN_TRAIN, EXP_R2_TRAIN_LABELED
    EXP_TOTAL = len(man)
    EXP_SUP = int(man["use_supervised"].sum())
    EXP_SUP_GT = int((man["target_source"] == "gt").sum())
    EXP_SUP_PSEUDO = int((man["target_source"] == "pseudo").sum())
    EXP_CONS = int(man["use_consistency"].sum())
    EXP_IGNORED = int(man["ignore_image"].sum())
    val_ids = set(pd.read_csv(CHAMP_VAL_CSV)["image_path"].astype(str))
    gt = man[man["target_source"] == "gt"]
    EXP_GT_IN_TRAIN = int((~gt["image_id"].astype(str).isin(val_ids)).sum())
    EXP_R2_TRAIN_LABELED = EXP_GT_IN_TRAIN + EXP_SUP_PSEUDO
    # cross-check against nb05b's independently written counts
    side = MANIFEST_DIR / "rebalance_summary.json"
    assert side.exists(), f"nb05b sidecar missing: {side}"
    s = json.loads(side.read_text())
    for k, v in (("n_total", EXP_TOTAL), ("n_supervised", EXP_SUP),
                 ("n_gt", EXP_SUP_GT), ("n_pseudo", EXP_SUP_PSEUDO),
                 ("n_consistency", EXP_CONS), ("n_ignored", EXP_IGNORED)):
        assert int(s[k]) == v, f"nb05b sidecar {k}={s[k]} != manifest {v}"
    # the whole point of the fork: the consistency pool must be champion-scale
    # (NOT the ~20k capped/round-1 scale). The A4C2 purge legitimately removes the
    # ~44k A4C2 wrong-domain frames from consistency, so it floors lower but is still
    # far above the cap scale -- relax the gate only when the sidecar declares the purge.
    _cons_floor = 50_000 if s.get("purge_a4c2") else 100_000
    assert EXP_CONS > _cons_floor, (
        f"consistency pool {EXP_CONS} below floor {_cons_floor} -- the cap leaked back in")
    print(f"[gate] derived: total={EXP_TOTAL} sup={EXP_SUP} (gt {EXP_SUP_GT} + "
          f"pseudo {EXP_SUP_PSEUDO}) cons={EXP_CONS} ign={EXP_IGNORED} "
          f"gt_in_train={EXP_GT_IN_TRAIN} r2_labeled={EXP_R2_TRAIN_LABELED}")


# =====================================================================
# per-sample-weighted labeled dataset + collate (change 3 plumbing)
# =====================================================================
class WeightedKeypointDataset(KeypointDataset):
    """KeypointDataset that also emits the per-sample manifest `sample_weight`
    (column on self.dataframe). Missing/NaN => 1.0 (byte-identical to the
    champion for any un-weighted row, e.g. val)."""

    def __getitem__(self, idx: int) -> dict:
        base = super().__getitem__(idx)
        w = 1.0
        try:
            val = self.dataframe.iloc[idx].get("sample_weight", 1.0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                w = float(val)
        except Exception:
            w = 1.0
        base["sample_weight"] = w
        return base


def weighted_keypoint_collate_fn(batch):
    out = keypoint_collate_fn(batch)
    out["sample_weight"] = torch.tensor(
        [float(it.get("sample_weight", 1.0)) for it in batch], dtype=torch.float32
    )
    return out


# =====================================================================
# ROUND-2 PATCHES = champion P1..P4,P6 (verbatim) + P0' + P5' + P_SUP
# =====================================================================
# P0' — repoint DATA_ROOT to the round-2 mirror (this is the ONLY change to P0).
_PATCH_DATA_R2 = ('DATA_ROOT_PATH = "data"', f'DATA_ROOT_PATH = {str(DATAROOT_R2)!r}')

# P5' — champion HC@700 backward, but PER-SAMPLE-WEIGHTED (change 3, HC leg).
#       anchor is the champion/nb04 cons-block header (identical to champion P5).
_PATCH_HC_BACKWARD_W = (
    CH._PATCH_HC_BACKWARD[0],
    """            # [R2 HC-RES700 SUP BACKWARD BEGIN] HC highres supervised backward,
            # drawn from the dedicated 700 loader, grad-accumulated into the SAME
            # outer batch. PER-SAMPLE WEIGHTED: per-sample MSE * manifest sample_weight,
            # then mean (round-2 change 3, HC leg). No-op when hc_lab_loader is None.
            if hc_lab_loader is not None:
                try:
                    _hl = next(hc_lab_iter)
                except StopIteration:
                    hc_lab_iter = iter(hc_lab_loader)
                    _hl = next(hc_lab_iter)
                _hl_imgs = _hl["image"].to(device)
                _hl_hm = torch.stack(_hl["heatmap"], 0).to(device)
                _hl_w = _hl["sample_weight"].to(device)
                _hl_logits = model(_hl_imgs, task_id="HC")
                _hl_per = ((torch.sigmoid(_hl_logits) - _hl_hm) ** 2).mean(dim=[1, 2, 3])
                _hl_loss = (_hl_per * _hl_w).mean()
                _hl_loss.backward()
                epoch_train_losses["HC"].append(float(_hl_loss.item()))
            # [R2 HC-RES700 SUP BACKWARD END]

            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — cons forward(s) +
            # backward(s) [grad-accumulating, λ-weighted].
            cons_batch_value = 0.0""",
)

# P_SUP — the mixed-518 batch-mean sup loss -> PER-SAMPLE-WEIGHTED (change 3).
_PATCH_SUP_WEIGHTED = (
    """                pred_logits = model(task_images, task_id=current_task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                sup_loss = F.mse_loss(pred_heatmaps, task_heatmaps)
                sup_loss.backward()""",
    """                pred_logits = model(task_images, task_id=current_task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                # [R2 WEIGHTED SUP] per-sample MSE * manifest sample_weight, then mean
                # (round-2 change 3). Weight 1.0 for any un-weighted sample => champion.
                _r2_w = batch["sample_weight"][task_indices].to(device)
                _r2_per = ((pred_heatmaps - task_heatmaps) ** 2).mean(dim=[1, 2, 3])
                sup_loss = (_r2_per * _r2_w).mean()
                sup_loss.backward()""",
)

# =====================================================================
# UMv2 — the ambiguous consistency pool goes from conf-only to FULL UniMatch v2
# =====================================================================
# v3 (LB 23.7828) runs conf-only: ONE strong-aug student stream against the EMA
# teacher's weak pseudo-label, masked per task by teacher peak height. Full
# UniMatch v2 (Yang2023 §3.2) adds two more student streams against the SAME
# teacher target and the SAME mask:
#   * dual-strong — a 2nd independent strong aug (s2)
#   * feature perturbation — channel dropout p=0.5 at the encoder output, on the
#     WEAK input (so the student sees the same image the teacher saw, perturbed
#     in feature space instead of pixel space)
#
# Both mechanisms already exist as single-variable forks of this same nb04
# (folder-02 nb05 = +FP, nb06 = +dual) and both REGRESSED there: nb05 18.7077
# and nb06 19.6528 vs nb04's 18.2524 DARK. That measurement does NOT settle the
# question here, for the following reasons:
#   * there the consistency pool was ALL 191k unlabeled frames, undifferentiated;
#     here it is the CURATED ambiguous bucket (131 330), with 15 856 outliers and
#     2 941 double-window frames already removed;
#   * nb05's regression was carried almost entirely by A4C (+3.21 of the +0.46);
#     A4C's pool was then 96.5% wrong-domain A4C2 and completely uncurated.
# So the population these streams consume is not the population that broke them.
#
# The helpers are the SHARED ones both forks used — nothing is reimplemented:
#   src.training.nb02_01_feature_perturbation.apply_feature_perturbation
#   src.training.nb02_00_unimatch_dataset.UnlabeledWeakStrongDataset(yield_two_strong=)
#
# cons_weight divides by N_STREAMS (3).
#
# WARNING (adversarial review, 2026-07-28): dividing by three equalizes the
# COEFFICIENT, not the gradient. In v3, `cons_loss` falls to about 2e-5, but the
# FP stream—with 50% of its channels zeroed—never converges to that level. Thus
# L_fp >> L_s1 and v4's TOTAL consistency gradient is strictly LARGER than v3's.
# This does not invalidate the run, but the result cannot be described as changing
# only the existence of the streams: consistency energy also increased. If v4
# wins, that remains a viable alternative explanation.
UMV2_FP_DROPOUT = 0.5      # FETUS-canonical (Echocare.py:314); FAILED NB01 used 0.3

_PATCH_UMV2_DATASET = (
    """        unl_dataset = UnlabeledWeakStrongDataset(
            data_root=_unl_data_root,
            weak_transform=weak_tf,
            strong_transform=strong_tf,
        )""",
    """        unl_dataset = UnlabeledWeakStrongDataset(
            data_root=_unl_data_root,
            weak_transform=weak_tf,
            strong_transform=strong_tf,
            # [UMV2] second independent strong aug per item (same transform,
            # fresh random params) -> unl_batch["image_strong2"].
            yield_two_strong=True,
        )""",
)

_PATCH_UMV2_WEIGHT = (
    """                        cons_weight = lam / n_cons_tasks_actual
                        cons_task_values = []""",
    """                        # [UMV2] 3 streams: strong(s1) + strong(s2) + FP(weak).
                        # Dividing keeps the total cons gradient per outer batch
                        # equal to v3's, so the A/B isolates the streams.
                        cons_weight = lam / (n_cons_tasks_actual * 3)
                        cons_task_values = []
                        strong2_all = unl_batch["image_strong2"].to(device, non_blocking=True)""",
)

_PATCH_UMV2_STREAMS = (
    """                            (cons_weight * cons_loss).backward()
                            cons_task_values.append(float(cons_loss.item()))
                            cons_backwards_total += 1""",
    """                            (cons_weight * cons_loss).backward()
                            cons_task_values.append(float(cons_loss.item()))
                            cons_backwards_total += 1

                            # [UMV2 STREAM 2] second independent strong aug.
                            # Same teacher target, same conf mask.
                            _s2_logits = model(strong2_all[sel], task_id=cons_task_id)
                            _cons_s2 = unimatch_consistency_loss(
                                student_strong_logits=_s2_logits,
                                teacher_weak_logits=teacher_logits_d,
                                confidence_mask=conf_mask,
                            )
                            (cons_weight * _cons_s2).backward()
                            cons_task_values.append(float(_cons_s2.item()))

                            # [UMV2 STREAM 3] feature perturbation on the WEAK
                            # input. encoder(weak) is a SECOND student forward —
                            # the teacher's pass is under no_grad and EMA-weighted,
                            # so it cannot be reused for student gradients.
                            # Single-site (encoder output): DINOv2 ViT-S is a flat
                            # transformer, FETUS's 6-site multi-stage FP does not
                            # port to it. bf16 autocast is applied here EXPLICITLY
                            # because this path calls encoder/head directly and so
                            # bypasses the MultiTaskModelFactory.forward wrapper.
                            with torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=_UMV2_BF16):
                                _wf = model.encoder(weak)
                                _wf = _apply_fp(_wf, p=_UMV2_FP_P, training=True)
                                _fp_logits = model.heads[cons_task_id](
                                    _wf, out_size=model.heatmap_size)
                            _fp_logits = _fp_logits.float()
                            _cons_fp = unimatch_consistency_loss(
                                student_strong_logits=_fp_logits,
                                teacher_weak_logits=teacher_logits_d,
                                confidence_mask=conf_mask,
                            )
                            (cons_weight * _cons_fp).backward()
                            cons_task_values.append(float(_cons_fp.item()))""",
)

_PATCH_UMV2_IMPORT = (
    """from src.training.nb02_01_per_task_confidence import (
    compute_per_task_thresholds,
    per_task_confidence_mask,
)""",
    """from src.training.nb02_01_per_task_confidence import (
    compute_per_task_thresholds,
    per_task_confidence_mask,
)
# [UMV2] FP operator — the SAME shared helper folder-02 nb05 used.
from src.training.nb02_01_feature_perturbation import (
    apply_feature_perturbation as _apply_fp,
)""",
)

_UMV2_PATCHES = [
    _PATCH_UMV2_IMPORT,
    _PATCH_UMV2_DATASET,
    _PATCH_UMV2_WEIGHT,
    _PATCH_UMV2_STREAMS,
]

# ordered patch list applied to the nb04 source
_ALL_PATCHES_R2 = [
    _PATCH_DATA_R2,          # P0'
    CH._PATCH_GLOBALS,       # P1 (verbatim)
    CH._PATCH_SMOKE_READ,    # P2 (verbatim)
    CH._PATCH_HC_LOADER,     # P3 (verbatim)
    CH._PATCH_SUP_SKIP,      # P4 (verbatim)
    _PATCH_HC_BACKWARD_W,    # P5' (weighted HC leg)
    CH._PATCH_SMOKE_LOOP,    # P6 (verbatim)
    _PATCH_SUP_WEIGHTED,     # P_SUP (weighted mixed-518 leg)
]


# =====================================================================
# round-2 dataroot builder (change 1 + change 2)
# =====================================================================
def _symlink(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            return  # a real dir/file already there; leave it
    dst.symlink_to(src)


def build_round2_dataroot(force: bool = False, verbose: bool = True):
    """Materialise `_dataroot_r2/` from the champion FLAT source + the nb05
    manifest. Returns a dict of built-artifact stats."""
    man = pd.read_csv(MANIFEST_CSV)
    assert man["image_id"].is_unique, "manifest image_id not unique"
    _derive_expected(man)  # NOCAP FORK: data-dependent gate, see _derive_expected

    # ---- curation integrity (manifest regime totals) ----
    n_sup = int(man["use_supervised"].sum())
    n_gt = int((man["target_source"] == "gt").sum())
    n_pseudo = int((man["target_source"] == "pseudo").sum())
    n_cons = int(man["use_consistency"].sum())
    n_ign = int(man["ignore_image"].sum())
    assert n_sup == EXP_SUP, f"supervised={n_sup} != {EXP_SUP}"
    assert n_gt == EXP_SUP_GT and n_pseudo == EXP_SUP_PSEUDO, (n_gt, n_pseudo)
    assert n_cons == EXP_CONS, f"consistency={n_cons} != {EXP_CONS}"
    assert n_ign == EXP_IGNORED, f"ignored={n_ign} != {EXP_IGNORED}"
    assert len(man) == EXP_TOTAL, f"total={len(man)} != {EXP_TOTAL}"

    w_by_id = man.set_index("image_id")["sample_weight"]

    # ---- dirs + symlinks ----
    (DATAROOT_R2 / "csv").mkdir(parents=True, exist_ok=True)
    (R2_UNL_ROOT / "csv_unlabeled").mkdir(parents=True, exist_ok=True)
    data_images = REPO / "data" / "images"
    data_images_unl = REPO / "data" / "images_unlabeled"
    _symlink(data_images, DATAROOT_R2 / "images")                    # GT rows
    _symlink(data_images_unl, DATAROOT_R2 / "images_unlabeled")      # pseudo GT rows
    _symlink(data_images_unl, R2_UNL_ROOT / "images_unlabeled")      # consistency corpus

    # ---- VAL: byte copy of champion internal_val.csv ----
    shutil.copyfile(CHAMP_VAL_CSV, R2_VAL_CSV)

    # =========================================================
    # change 1: round2_train_labeled.csv = GT-in-train + safe-pseudo
    # =========================================================
    gt_ids = set(man.loc[man["target_source"] == "gt", "image_id"].astype(str))

    # (a) GT rows: champion TRAIN rows whose image_path IS a manifest-GT id.
    tr = pd.read_csv(CHAMP_TRAIN_CSV)
    tr_paths = tr["image_path"].astype(str)
    gt_mask = tr_paths.isin(gt_ids)
    gt_rows = tr[gt_mask].copy()
    gt_rows["sample_weight"] = gt_rows["image_path"].astype(str).map(w_by_id).astype(float)
    assert gt_rows["sample_weight"].notna().all(), "some GT-in-train row missing a weight"
    assert len(gt_rows) == EXP_GT_IN_TRAIN, f"GT-in-train={len(gt_rows)} != {EXP_GT_IN_TRAIN}"
    n_dropped_train = int((~gt_mask).sum())  # champion-train rows not in the manifest (21 femur)

    # (b) PSEUDO rows: safe-unlabeled images with denormalized pseudo landmarks.
    pz = np.load(PSEUDO_NPZ)
    nb00 = pd.read_csv(NB00_MANIFEST).set_index("image_id")
    ps = man[man["target_source"] == "pseudo"]
    max_pts = max(NUM_CLASSES.values())  # 22 (PLAX)
    point_cols = [f"point_{i}_xy" for i in range(1, max_pts + 1)]
    pseudo_records = []
    for _, r in ps.iterrows():
        iid = str(r["image_id"]); tid = str(r["task_id"])
        k = NUM_CLASSES[tid]
        vec = pz[iid]
        assert vec.shape[0] == 2 * k, f"{iid} vec {vec.shape[0]} != 2*{k}"
        W = float(nb00.loc[iid, "image_width"]); H = float(nb00.loc[iid, "image_height"])
        rec = {c: np.nan for c in point_cols}
        for i in range(k):
            x = float(vec[2 * i]) * W
            y = float(vec[2 * i + 1]) * H
            rec[f"point_{i + 1}_xy"] = f"[{round(x, 4)}, {round(y, 4)}]"
        rec["image_path"] = "images_unlabeled/" + iid  # resolves via data_root/images_unlabeled
        rec["task_id"] = tid
        rec["task_name"] = "Regression"
        rec["num_classes"] = int(k)
        rec["height"] = int(round(H))
        rec["width"] = int(round(W))
        rec["sample_weight"] = float(r["sample_weight"]) * float(LAMBDA_PSEUDO)
        pseudo_records.append(rec)
    pseudo_rows = pd.DataFrame(pseudo_records)
    assert len(pseudo_rows) == EXP_SUP_PSEUDO, f"pseudo={len(pseudo_rows)} != {EXP_SUP_PSEUDO}"

    r2_labeled = pd.concat([gt_rows, pseudo_rows], ignore_index=True)
    assert len(r2_labeled) == EXP_R2_TRAIN_LABELED, (
        f"round2 labeled={len(r2_labeled)} != {EXP_R2_TRAIN_LABELED}")
    # no path collides between train-labeled and val (train/val disjoint)
    val_paths = set(pd.read_csv(R2_VAL_CSV)["image_path"].astype(str))
    assert not (set(r2_labeled["image_path"].astype(str)) & val_paths), \
        "round2 labeled overlaps val -> train-on-val leak"
    r2_labeled.to_csv(R2_TRAIN_CSV, index=False)

    # =========================================================
    # change 2: consistency pool = ambiguous ids only (subset champion manifests)
    # =========================================================
    amb_ids = set(man.loc[man["use_consistency"], "image_id"].astype(str))
    unl_kept = 0
    per_csv = {}
    for src_csv in sorted(CHAMP_UNL_CSV_DIR.glob("*.csv")):
        df = pd.read_csv(src_csv)
        sub = df[df["image_path"].astype(str).isin(amb_ids)].reset_index(drop=True)
        sub.to_csv(R2_UNL_ROOT / "csv_unlabeled" / src_csv.name, index=False)
        per_csv[src_csv.name] = len(sub)
        unl_kept += len(sub)
    assert unl_kept == EXP_CONS, f"consistency subset={unl_kept} != {EXP_CONS}"

    stats = dict(
        manifest_supervised=n_sup, manifest_gt=n_gt, manifest_pseudo=n_pseudo,
        manifest_consistency=n_cons, manifest_ignored=n_ign, manifest_total=len(man),
        r2_train_labeled=len(r2_labeled), gt_in_train=len(gt_rows),
        pseudo_rows=len(pseudo_rows), gt_in_val_excluded=n_gt - len(gt_rows),
        champ_train_rows_not_in_manifest=n_dropped_train,
        consistency_pool=unl_kept, consistency_per_task=per_csv,
        val_rows=len(pd.read_csv(R2_VAL_CSV)),
        weight_gt_hist={float(k): int(v) for k, v in
                        gt_rows["sample_weight"].value_counts().sort_index().items()},
        pseudo_weight_min=float(pseudo_rows["sample_weight"].min()),
        pseudo_weight_max=float(pseudo_rows["sample_weight"].max()),
        dataroot=str(DATAROOT_R2),
    )
    if verbose:
        print("[build_round2_dataroot] " + json.dumps(stats, indent=2))
    return stats


# =====================================================================
# preflight
# =====================================================================
def preflight(build: bool = True):
    print("=" * 84)
    print("PRE-FLIGHT — res700-flat champion recipe, round-2 curation manifest (3 changes)")
    print("=" * 84)

    src = NB04_SRC.read_text()
    for i, (anchor, _) in enumerate(_ALL_PATCHES_R2):
        assert anchor in src, f"G0 FAIL: patch anchor P{i} missing:\n{anchor[:80]!r}"
        assert src.count(anchor) == 1, f"G0 FAIL: anchor P{i} not unique ({src.count(anchor)}x)"
    print(f"  G0 all {len(_ALL_PATCHES_R2)} patch anchors found & unique in {NB04_SRC.name}  OK")

    # champion HC@700 machinery re-used verbatim (P1..P4,P6)
    assert _ALL_PATCHES_R2[1] == CH._PATCH_GLOBALS
    assert _ALL_PATCHES_R2[2] == CH._PATCH_SMOKE_READ
    assert _ALL_PATCHES_R2[3] == CH._PATCH_HC_LOADER
    assert _ALL_PATCHES_R2[4] == CH._PATCH_SUP_SKIP
    assert _ALL_PATCHES_R2[6] == CH._PATCH_SMOKE_LOOP
    assert _PATCH_HC_BACKWARD_W[0] == CH._PATCH_HC_BACKWARD[0]  # same anchor, weighted body
    print("  G_HC P1..P4,P6 byte-identical to hc_res700_flat_engine; P5' shares anchor  OK")

    if build:
        stats = build_round2_dataroot(verbose=False)
    else:
        stats = None

    # pool glob = exactly the 2 round-2 CSVs
    files = sorted(Path(DATAROOT_R2 / "csv").glob("*.csv"))
    names = [f.name for f in files]
    assert names == ["internal_val.csv", "round2_train_labeled.csv"], names
    print(f"  G1 _dataroot_r2/csv globs to {names}  OK")

    # HC@700 divisibility + stock cold-start (champion parity)
    # HC_INPUT_SIZE is None => the HC@700 leg is OFF and HC rides the mixed-518 batch.
    # The champion engine already declares None legal (hc_res700_flat_engine.py:410
    # `assert HC_INPUT_SIZE is None or int(HC_INPUT_SIZE) % 14 == 0`); this clone had
    # dropped that half of the guard, which made the OFF case unreachable. Restoring it
    # is a no-op for any integer value, so parity with 700 is unaffected.
    assert HC_INPUT_SIZE is None or HC_INPUT_SIZE % 14 == 0, HC_INPUT_SIZE
    ns_probe = {}
    for line in src.splitlines():
        if line.startswith("ENCODER ") or line.startswith("ENCODER_WEIGHTS "):
            exec(line, ns_probe)  # noqa: S102
    assert ns_probe.get("ENCODER") == "vit_small_patch14_dinov2.lvd142m"
    assert ns_probe.get("ENCODER_WEIGHTS") == "pretrained"
    _hc_leg = ("OFF (HC rides the mixed-518 batch)" if HC_INPUT_SIZE is None
               else f"@{HC_INPUT_SIZE} (grid {HC_INPUT_SIZE // 14})")
    print(f"  G2 encoder stock cold-start; HC leg {_hc_leg}  OK")

    if stats is not None:
        print("  G3 curation build:")
        print("     " + json.dumps({k: stats[k] for k in (
            "manifest_supervised", "r2_train_labeled", "gt_in_train", "pseudo_rows",
            "consistency_pool", "val_rows", "weight_gt_hist")}, indent=6).replace("\n", "\n     "))

    print("=" * 84)
    print("PRE-FLIGHT PASSED\n")
    return stats


# =====================================================================
# patched-module builder + injections
# =====================================================================
# Hook for downstream forks that need EXTRA (anchor, replacement) pairs applied
# on top of this engine's own patch set, without copying the builder. Empty by
# default, so v3/v4/v6 behaviour is byte-identical. Used by
# experiments/09_M2_VITB (encoder swap ViT-S -> ViT-B+reg4).
EXTRA_PATCHES: list = []


def _build_patched_module():
    src = NB04_SRC.read_text()
    patches = (list(_ALL_PATCHES_R2) + (_UMV2_PATCHES if USE_UMV2 else [])
               + list(EXTRA_PATCHES))
    for i, (anchor, repl) in enumerate(patches):
        if src.count(anchor) != 1:
            raise RuntimeError(
                f"patch anchor P{i} matched {src.count(anchor)}x (need exactly 1): "
                f"{anchor[:80]!r}")
        src = src.replace(anchor, repl, 1)
    ns = {"__name__": "nb04_r2_patched", "__file__": str(NB04_SRC)}
    exec(compile(src, str(NB04_SRC), "exec"), ns)  # noqa: S102
    assert ns["DATA_ROOT_PATH"] == str(DATAROOT_R2), "P0' data patch did not take"
    assert "HC_INPUT_SIZE" in ns, "P1 global patch did not take"
    # NAMESPACE INJECTION — swap the labeled dataset class + collate to weighted
    # variants (late binding: main() resolves these globals at call time). This
    # threads sample_weight through the mixed-518 loader AND the HC@700 loader.
    ns["KeypointDataset"] = WeightedKeypointDataset
    ns["keypoint_collate_fn"] = weighted_keypoint_collate_fn
    if USE_BF16:
        _wrap_forward_bf16(ns)
    if USE_UMV2:
        # the FP stream reads these from the exec'd namespace
        ns["_UMV2_BF16"] = bool(USE_BF16)
        ns["_UMV2_FP_P"] = float(UMV2_FP_DROPOUT)
        assert "_apply_fp" in ns, "UMv2 import patch did not take"
        print(f"[UMv2] ambiguous pool = FULL UniMatch v2: strong(s1) + strong(s2) "
              f"+ FP(weak, p={UMV2_FP_DROPOUT}); cons_weight /= 3 so the total "
              f"consistency gradient matches v3")
    return ns


USE_BF16 = True   # --no-bf16 restores the fp32 recipe v2 ran
USE_UMV2 = False  # --umv2 turns the ambiguous pool into full UniMatch v2
LR_REF_BATCH = None  # --lr-ref-batch N: scale LRs as though the batch were N


def _wrap_forward_bf16(ns) -> None:
    """Run the encoder+head matmuls in bf16 while TRAINING; everything else stays
    fp32.

    Why: the recipe is pure fp32 (nb04 has no autocast/GradScaler) and this card
    is power-limited (measured 299.4 W of a 300 W cap at 86 C), so fp32 tensor
    throughput — not occupancy — is the wall. Measured on the real model, batch
    32, 518px, labeled+UniMatch step:
        fp32   2432 ms/step   13.2 img/s   44.7 GiB
        bf16    697 ms/step   45.9 img/s   26.1 GiB
    Batch size does NOT help here (12.1 -> 13.3 img/s from batch 4 to 64), so
    this is the only lever on wall-clock that is not a data or schedule change.

    Scope is deliberately narrow:
      * the autocast region covers ONLY the model forward, and the output is cast
        back to fp32 on the way out. Every downstream op the recipe cares about —
        sigmoid, MSE, the UniMatch consistency loss, the confidence gate, DARK —
        therefore sees exactly the fp32 tensors it saw before;
      * master weights and the AdamW states stay fp32 (autocast never touches
        parameters), so the optimizer maths is unchanged;
      * `self.training` gates it, so validation and best-checkpoint selection run
        in fp32 and the val MRE stays numerically comparable to the v2 run.
    """
    MF = ns["MultiTaskModelFactory"]
    if getattr(MF, "_bf16_wrapped", False):
        return
    _orig = MF.forward

    def _fwd(self, *a, **kw):
        if not self.training:
            return _orig(self, *a, **kw)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _orig(self, *a, **kw)
        if torch.is_tensor(out):
            return out.float()
        if isinstance(out, (tuple, list)):
            return type(out)(o.float() if torch.is_tensor(o) else o for o in out)
        return out

    MF.forward = _fwd
    MF._bf16_wrapped = True
    print("[bf16] model forward wrapped in bfloat16 autocast (train only; "
          "outputs cast back to fp32; val/eval stay fp32)")


BASE_BATCH = 4          # the champion recipe's BATCH_SIZE (nb04 source, line 87)
BASE_S1_HEAD_LR = 1e-3
BASE_S2_ENC_LR = 1e-5
BASE_S2_HEAD_LR = 5e-5


def _scaled_lrs(batch_size: int):
    """Scale the champion LRs for a larger batch, sqrt rule (AdamW).

    Explicit deviation from champion parity, requested during the 2026-07-27 review: the
    recipe trains at BATCH_SIZE=4 on a 96 GB card that peaks near 1.4 GB, so the
    35-epoch stage-2 runs ~31 h for no reason. Raising the batch alone would be
    worse than useless -- fewer, larger steps at an LR tuned for batch 4 under-trains,
    and the run would answer "was the LR too small" instead of "does curation work".

    sqrt rather than linear: linear scaling is the SGD rule; AdamW already normalises
    by the gradient second moment, so linear over-shoots on a fine-tune whose encoder
    LR is 1e-5. sqrt(k) is the conservative choice when the goal is to preserve the
    optimisation regime, not to squeeze the last epoch of accuracy.

    NOTE this breaks byte-parity with the res700 champion. A win here is NOT
    attributable to curation alone -- batch+LR is now a 5th changed variable."""
    k = batch_size / BASE_BATCH
    s = k ** 0.5
    return dict(k=k, scale=s,
                s1_head_lr=BASE_S1_HEAD_LR * s,
                s2_enc_lr=BASE_S2_ENC_LR * s,
                s2_head_lr=BASE_S2_HEAD_LR * s)


def run(tag: str, s1_epochs: int, s2_epochs: int, smoke_steps: int = 0,
        warm_start_ckpt: str | None = None, batch_size: int = BASE_BATCH):
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    stats = preflight(build=True)

    if smoke_steps and smoke_steps > 0:
        os.environ["HC_SMOKE_STEPS"] = str(int(smoke_steps))
    else:
        os.environ.pop("HC_SMOKE_STEPS", None)

    s1_ckpt = out / "stage1_heads_only.pth"
    s2_ckpt = out / "stage2_unfreeze.pth"
    ns = _build_patched_module()
    main = ns["main"]

    # ===== STAGE 1 — LP heads-only, encoder FROZEN (stock or warm), HC leg @700 =====
    # BATCH_SIZE lives in the exec'd namespace, so this one assignment reaches BOTH
    # the mixed-518 supervised loader and the HC@700 dedicated leg (the champion
    # HC patch reads the same global). The unlabeled loader keeps the 1:1
    # labeled:unlabeled ratio the champion runs at.
    # LR_REF_BATCH decouples the LR from the batch. Default None = scale off the
    # real batch (the original behaviour).
    #
    # MEASURED 2026-07-28: at batch 96 the sqrt rule gives x4.90 (s1_head 4.90e-3,
    # s2_enc 4.90e-5) and stage-1 BREAKS. Trajectory over its 5 epochs:
    #     v4 @x4.90   203.48 -> 94.86 -> 86.14 -> 82.09 -> 81.83   (plateau, last Δ 0.25)
    #     v3 @x2.83    79.31 -> 61.79 -> 53.98 -> 46.69 -> 44.38   (still falling, Δ 2.31)
    # Stage-1 runs with the encoder FROZEN and UniMatch OFF, so nothing but
    # batch+LR can explain it. The run was killed and relaunched with the batch
    # requested batch size but with the learning rates pinned to v3's, so the comparison is about
    # UMv2 instead of about a crippled initialisation.
    _lr_batch = int(LR_REF_BATCH) if LR_REF_BATCH else int(batch_size)
    LRS = _scaled_lrs(_lr_batch)
    ns["BATCH_SIZE"] = int(batch_size)
    print(f"\n[batch] BATCH_SIZE {BASE_BATCH} -> {batch_size}  (x{batch_size/BASE_BATCH:.0f})")
    print(f"[batch] LR scaled as though the batch were {_lr_batch} (x{LRS['scale']:.2f}): "
          f"s1_head {LRS['s1_head_lr']:.2e}  s2_enc {LRS['s2_enc_lr']:.2e}  "
          f"s2_head {LRS['s2_head_lr']:.2e}")
    if _lr_batch != batch_size:
        print(f"[batch] LR DECOUPLED from the batch intentionally — sqrt scaling for "
              f"batch {batch_size} would give x{(batch_size/BASE_BATCH)**0.5:.2f}, which "
              f"broke stage 1 (81.83 vs v3's 44.38).")
    if batch_size != BASE_BATCH:
        print("[batch] WARNING: byte-parity with the res700 champion is BROKEN by "
              "this change; batch+LR is a 5th variable on top of the 4 curation ones.")

    ns["NUM_EPOCHS"] = s1_epochs
    ns["HC_INPUT_SIZE"] = HC_INPUT_SIZE
    print(f"\n===== STAGE 1 (LP heads-only) {s1_epochs} ep  init={warm_start_ckpt or 'STOCK'} =====")
    t0 = time.time()
    r1 = main(leaky=False, train_split_csv=str(R2_TRAIN_CSV), val_split_csv=str(R2_VAL_CSV),
              preprocess=PREPROCESS, freeze_encoder=True, head_lr=LRS["s1_head_lr"],
              init_from_ckpt=warm_start_ckpt,   # None => stock cold-start (champion parity)
              ema_decay=None, enable_unimatch=False,
              enable_confidence=True, conf_quantile=0.25,
              model_save_path=str(s1_ckpt))
    s1_wall = round(time.time() - t0, 1)
    print(f"[stage1] best_val_mre={r1.get('best_val_mre')}  wall={s1_wall}s")

    # ===== STAGE 2 — FT unfreeze + UniMatch(ambiguous) + EMA, init from OUR stage1 =====
    ns["NUM_EPOCHS"] = s2_epochs
    ns["HC_INPUT_SIZE"] = HC_INPUT_SIZE
    print(f"\n===== STAGE 2 (FT unfreeze + UniMatch + EMA) {s2_epochs} ep =====")
    t0 = time.time()
    r2 = main(leaky=False, train_split_csv=str(R2_TRAIN_CSV), val_split_csv=str(R2_VAL_CSV),
              preprocess=PREPROCESS, freeze_encoder=False, init_from_ckpt=str(s1_ckpt),
              encoder_lr=LRS["s2_enc_lr"], head_lr=LRS["s2_head_lr"], ema_decay=0.999,
              enable_unimatch=True, lambda_cons_max=0.5, lambda_ramp_epochs=7,
              unlabeled_batch_size=int(batch_size),   # keep the champion's 1:1 ratio
              num_workers_unlabeled=8,                # 4 starves the bigger batch
              unlabeled_data_root=str(R2_UNL_ROOT),
              unlabeled_weak_preprocess="ssl_weak_v1",
              unlabeled_strong_preprocess="ssl_strong_v1",
              enable_confidence=True, conf_quantile=0.25,
              model_save_path=str(s2_ckpt))
    s2_wall = round(time.time() - t0, 1)

    summary = dict(
        tag=tag,
        kind="res700-flat champion recipe + NOCAP round-2 curation manifest (nb05b)",
        changes=[
            f"1: weighted+augmented labeled set (GT-in-train {EXP_GT_IN_TRAIN} + "
            f"budgeted pseudo {EXP_SUP_PSEUDO} = {EXP_R2_TRAIN_LABELED}; per-sample "
            "sample_weight; pseudo target = denorm pseudo landmarks, λp=1.0)",
            f"2: NOCAP -- consistency pool = {EXP_CONS} images (v1 round-2 was 13819; "
            f"champion 191170). nb00 ran with --per-task-unlabeled-cap 0.",
            "3: per-sample-weighted supervised MSE (mixed-518 + HC@700 legs), λ_cons=0.5 unchanged",
            "4: nb05b deviations vs v1 -- see runs/05b_*/rebalance_summary.json "
            "(pseudo budget drawn from full corpus; IVC bypasses curation)",
        ],
        manifest_dir=str(MANIFEST_DIR),
        init=(warm_start_ckpt or "STOCK timm DINOv2 cold-start (champion parity)"),
        encoder="vit_small_patch14_dinov2.lvd142m", preprocess=PREPROCESS,
        hc_input_size=HC_INPUT_SIZE, lambda_pseudo=LAMBDA_PSEUDO,
        smoke_steps=int(smoke_steps) if smoke_steps else 0,
        s1_epochs=s1_epochs, s2_epochs=s2_epochs,
        stage1_best_val=r1.get("best_val_mre"), stage1_wall_sec=s1_wall,
        stage2_best_val=r2.get("best_val_mre"), stage2_wall_sec=s2_wall,
        stage2_confidence_thresholds=r2.get("confidence_thresholds"),
        stage2_cons_loss_per_epoch=r2.get("cons_loss_per_epoch"),
        stage1_ckpt=str(s1_ckpt), stage2_ckpt=str(s2_ckpt),
        build_stats=stats,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {out / 'summary.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1+1 epoch smoke (step-capped)")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="just materialise _dataroot_r2")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke-steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH,
                    help="champion recipe is 4; larger breaks parity but the LRs "
                         "are sqrt-scaled to match (see _scaled_lrs)")
    ap.add_argument("--warm-start-ckpt", type=str, default=None,
                    help="FLAG: default None = stock cold-start (champion parity). "
                         "Pass a champion stage2 ckpt to warm-start instead.")
    ap.add_argument("--no-bf16", action="store_true",
                    help="disable the bf16 autocast on the training forward and "
                         "run the fp32 recipe v2 ran (~3.5x slower, measured)")
    ap.add_argument("--lr-ref-batch", type=int, default=None,
                    help="scale LRs as though the batch were N, independently of "
                         "the actual --batch-size. With batch 96, sqrt gives x4.90 "
                         "and breaks stage 1 (measured: 81.83 vs 44.38); "
                         "--lr-ref-batch 32 preserves the v3 LRs.")
    ap.add_argument("--umv2", action="store_true",
                    help="ambiguous consistency pool: conf-only -> FULL UniMatch v2 "
                         "(dual-strong + feature perturbation, same teacher target "
                         "and mask, cons_weight/3)")
    a = ap.parse_args()
    USE_BF16 = not a.no_bf16
    USE_UMV2 = bool(a.umv2)
    LR_REF_BATCH = a.lr_ref_batch
    if a.build_only:
        build_round2_dataroot(verbose=True)
        sys.exit(0)
    if a.preflight_only:
        preflight(build=True)
        sys.exit(0)
    if a.smoke:
        run(a.tag or "00_train_r2_SMOKE_v1_0724", s1_epochs=1, s2_epochs=1,
            smoke_steps=int(a.smoke_steps), warm_start_ckpt=a.warm_start_ckpt,
            batch_size=int(a.batch_size))
    else:
        run(a.tag or "01_train_r2_v1_0724", s1_epochs=5, s2_epochs=35,
            smoke_steps=0, warm_start_ckpt=a.warm_start_ckpt,
            batch_size=int(a.batch_size))
