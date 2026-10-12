"""run_nb04v3_a4c_no_ssl.py — M1-R3 driver (2-stage LP-FT chain).

Lives in experiments/experiments_alphonse/. ALL outputs (run logs, config
snapshots, checkpoints) are written INSIDE this folder. Shared READ-ONLY inputs
(SPLITS, the pinned unlabeled corpus, baseline + _shared_src code) are read from
the repo root but never modified.

Clone of the nb04 conf-only 2-stage chain with EXACTLY ONE variable flipped:
A4C is removed from the consistency (UniMatch) loss —
exclude_cons_tasks=("A4C",). The quantile gate (q25), preproc, splits,
schedule, EMA, lambda-ramp, epochs, LRs are byte-identical to nb04. A4C is
STILL trained on its real labels (supervised MSE); only its self-pseudo-labeled
consistency gradient is dropped.

WHY: A4C is the single task whose pseudo-labels are net-negative (lowest
teacher peaks => q25 quantile lowers tau_A4C~0.071 => ~100% of A4C garbage
pseudo-labels pass => confirmation bias). The honest gate (nb04v2, global
tau_abs=0.5) confirmed it (A4C -0.88) but a single global floor over-gated
IVC (+0.95) and could not help PSAX => net +0.16. R3 isolates the A4C win
without touching any other task.

A/B anchor : plain nb04 conf-only leader, DARK-518 18.2524.
One knob   : exclude_cons_tasks () -> ("A4C",).
Predicted  : A4C ~27.5 or better, IVC/PSAX/rest flat (+/-0.1), mean ~18.15.
Falsified  : mean >= 18.2524. PSAX is NOT expected to move.

CORPUS HYGIENE (single-variable): unlabeled corpus pinned to
data/_corpus_nb04v2_no_valorg (8 task CSVs, NO VAL_ORG) — IDENTICAL to the
honest gate, so the A/B is clean against the SAME 18.2524 anchor.

Launch (manual):
  cd "$FU_BIOMETRY_ROOT"  # Set FU_BIOMETRY_ROOT to the repository root.
  nohup "${PYTHON:-python3}" \
      experiments/experiments_alphonse/run_nb04v3_a4c_no_ssl.py \
      > experiments/experiments_alphonse/runs/04v3_a4c_no_ssl_v1_0628/chain.log 2>&1 &
"""
import io
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
ALF_DIR = _here                              # experiments/experiments_alphonse
REPO = _here
while not (REPO / "pyproject.toml").exists():
    if REPO == REPO.parent:
        raise RuntimeError("repo root not found (need pyproject.toml)")
    REPO = REPO.parent
import os
os.chdir(REPO)  # engine uses relative DATA_ROOT_PATH="data" / SPLITS paths

BASELINE_DIR = REPO / "Foundation-Model-for-Ultrasound-Biometry" / "baseline"
# engine lives HERE in experiments_alphonse; shared code read from repo root
for _p in (
    str(ALF_DIR),
    str(REPO / "experiments" / "_shared_src"),
    str(BASELINE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml
import nb04v3_train_a4c_no_ssl as train_module

# ---- pinned config (identical to nb04 except exclude_cons_tasks) ----
PREPROCESS_LABELED = "clahe_clip_sharpen_speckle"
PREPROCESS_UNLABELED_WEAK = "ssl_weak_v1"
PREPROCESS_UNLABELED_STRONG = "ssl_strong_v1"
EMA_DECAY_S1, EMA_DECAY_S2 = None, 0.999
UNIMATCH_S1, UNIMATCH_S2 = False, True
LAMBDA_CONS_MAX, LAMBDA_RAMP_EPOCHS = 0.5, 7
UNLABELED_BATCH_SIZE, NUM_WORKERS_UNLABEL = 4, 4
ENABLE_CONFIDENCE = True          # quantile q25 gate — same as nb04 leader
CONF_QUANTILE = 0.25              # nb04 default; unchanged

# === the ONE R3 variable ===
EXCLUDE_CONS_TASKS = ("A4C",)

# pinned clean corpus (no VAL_ORG) — single-variable hygiene (read-only input)
UNLABELED_DATA_ROOT = str(REPO / "data" / "_corpus_nb04v2_no_valorg")

STAGE1_NAME = "04v3_a4c_no_ssl_stage1_heads_only"
STAGE1_EPOCHS, STAGE1_HEAD_LR = 5, 1e-3
STAGE2_NAME = "04v3_a4c_no_ssl_stage2_unfreeze"
STAGE2_EPOCHS, STAGE2_ENC_LR, STAGE2_HEAD_LR = 35, 1e-5, 5e-5

OVERWRITE, LEAKY = False, False

# ---- ALL outputs inside experiments_alphonse ----
RUN_ROOT = ALF_DIR / "runs" / "04v3_a4c_no_ssl_v1_0628"
CKPT_DIR = ALF_DIR / "checkpoints"
STAGE1_RUN_DIR = RUN_ROOT / STAGE1_NAME
STAGE2_RUN_DIR = RUN_ROOT / STAGE2_NAME
STAGE1_CKPT = CKPT_DIR / (STAGE1_NAME + ".pth")
STAGE2_CKPT = CKPT_DIR / (STAGE2_NAME + ".pth")

for _ck in (STAGE1_CKPT, STAGE2_CKPT):
    if _ck.exists() and not OVERWRITE:
        raise RuntimeError(f"{_ck} exists — set OVERWRITE=True or change name.")
for _d in (STAGE1_RUN_DIR, STAGE2_RUN_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _run_stage(run_dir, fn):
    logf = open(run_dir / "train.log", "w", buffering=1)
    prev_out, prev_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(prev_out, logf), _Tee(prev_err, logf)
    t0 = time.time()
    try:
        return fn(), round(time.time() - t0, 1)
    finally:
        sys.stdout, sys.stderr = prev_out, prev_err
        logf.flush()
        logf.close()


# ---- config snapshots ----
cfg_stage1 = {
    "run_name": STAGE1_NAME, "stage": 1, "leaky": LEAKY,
    "train_split_csv": "experiments/SPLITS/train.csv",
    "val_split_csv": "experiments/SPLITS/internal_val.csv",
    "model_save_path": str(STAGE1_CKPT), "preprocess": PREPROCESS_LABELED,
    "freeze_encoder": True, "head_lr": STAGE1_HEAD_LR,
    "ema_decay": EMA_DECAY_S1, "enable_unimatch": UNIMATCH_S1,
    "num_epochs": STAGE1_EPOCHS,
    "engine": "nb04v3_train_a4c_no_ssl.py (stage1 == nb04 stage1, byte-equiv)",
}
cfg_stage2 = {
    "run_name": STAGE2_NAME, "stage": 2, "leaky": LEAKY,
    "train_split_csv": "experiments/SPLITS/train.csv",
    "val_split_csv": "experiments/SPLITS/internal_val.csv",
    "model_save_path": str(STAGE2_CKPT), "preprocess": PREPROCESS_LABELED,
    "freeze_encoder": False, "init_from_ckpt": str(STAGE1_CKPT),
    "encoder_lr": STAGE2_ENC_LR, "head_lr": STAGE2_HEAD_LR,
    "ema_decay": EMA_DECAY_S2, "enable_unimatch": UNIMATCH_S2,
    "lambda_cons_max": LAMBDA_CONS_MAX, "lambda_ramp_epochs": LAMBDA_RAMP_EPOCHS,
    "unlabeled_batch_size": UNLABELED_BATCH_SIZE,
    "num_workers_unlabeled": NUM_WORKERS_UNLABEL,
    "unlabeled_data_root": UNLABELED_DATA_ROOT,
    "unlabeled_weak_preprocess": PREPROCESS_UNLABELED_WEAK,
    "unlabeled_strong_preprocess": PREPROCESS_UNLABELED_STRONG,
    "enable_confidence": ENABLE_CONFIDENCE, "conf_quantile": CONF_QUANTILE,
    # === R3 variable ===
    "exclude_cons_tasks": list(EXCLUDE_CONS_TASKS),
    "num_epochs": STAGE2_EPOCHS,
    "engine": "nb04v3_train_a4c_no_ssl.py (R3: A4C removed from consistency loss)",
    "ab": {"anchor": "plain nb04 conf-only 18.2524 DARK-518",
           "one_knob": "exclude_cons_tasks () -> (A4C,)"},
}
(STAGE1_RUN_DIR / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_stage1, sort_keys=True))
(STAGE2_RUN_DIR / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_stage2, sort_keys=True))
print("config snapshots written.")
print(f"EXCLUDE_CONS_TASKS={EXCLUDE_CONS_TASKS}  gate=quantile(q{CONF_QUANTILE})  "
      f"corpus={UNLABELED_DATA_ROOT}")
print(f"outputs -> {RUN_ROOT}  ckpts -> {CKPT_DIR}")

# ---- STAGE 1 (frozen heads, EMA off, UniMatch off — byte-equiv nb04 stage1) ----
train_module.NUM_EPOCHS = STAGE1_EPOCHS
result_s1, wall1 = _run_stage(STAGE1_RUN_DIR, lambda: train_module.main(
    leaky=LEAKY,
    train_split_csv=cfg_stage1["train_split_csv"],
    val_split_csv=cfg_stage1["val_split_csv"],
    model_save_path=cfg_stage1["model_save_path"],
    preprocess=cfg_stage1["preprocess"],
    freeze_encoder=True, head_lr=STAGE1_HEAD_LR,
    ema_decay=EMA_DECAY_S1, enable_unimatch=UNIMATCH_S1,
))
(STAGE1_RUN_DIR / "summary.json").write_text(json.dumps(
    {**result_s1, "run_tag": STAGE1_NAME, "wall_time_sec": wall1,
     "checkpoint": str(STAGE1_CKPT)}, indent=2))
print("=== STAGE 1 DONE ===")

# ---- STAGE 2 (unfreeze, EMA + UniMatch + quantile gate, A4C cons removed) ----
train_module.NUM_EPOCHS = STAGE2_EPOCHS
result_s2, wall2 = _run_stage(STAGE2_RUN_DIR, lambda: train_module.main(
    leaky=LEAKY,
    train_split_csv=cfg_stage2["train_split_csv"],
    val_split_csv=cfg_stage2["val_split_csv"],
    model_save_path=cfg_stage2["model_save_path"],
    preprocess=cfg_stage2["preprocess"],
    freeze_encoder=False, init_from_ckpt=str(STAGE1_CKPT),
    encoder_lr=STAGE2_ENC_LR, head_lr=STAGE2_HEAD_LR,
    ema_decay=EMA_DECAY_S2, enable_unimatch=UNIMATCH_S2,
    lambda_cons_max=LAMBDA_CONS_MAX, lambda_ramp_epochs=LAMBDA_RAMP_EPOCHS,
    unlabeled_batch_size=UNLABELED_BATCH_SIZE,
    num_workers_unlabeled=NUM_WORKERS_UNLABEL,
    unlabeled_data_root=UNLABELED_DATA_ROOT,
    unlabeled_weak_preprocess=PREPROCESS_UNLABELED_WEAK,
    unlabeled_strong_preprocess=PREPROCESS_UNLABELED_STRONG,
    enable_confidence=ENABLE_CONFIDENCE, conf_quantile=CONF_QUANTILE,
    # === R3 variable ===
    exclude_cons_tasks=EXCLUDE_CONS_TASKS,
))
(STAGE2_RUN_DIR / "summary.json").write_text(json.dumps(
    {**result_s2, "run_tag": STAGE2_NAME, "wall_time_sec": wall2,
     "checkpoint": str(STAGE2_CKPT)}, indent=2))
print("=== STAGE 2 DONE ===")
print(json.dumps({
    "best_val_mre": result_s2.get("best_val_mre"),
    "exclude_cons_tasks": list(EXCLUDE_CONS_TASKS),
}, indent=2))
print(f"\nNEXT: DARK-518 + orig-px eval on {STAGE2_CKPT}, A/B vs plain nb04 18.2524.")
