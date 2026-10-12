"""_run_usfm_both.py — headless runner for the USFM supervised baseline.

Runs the folder-07 notebook logic end-to-end for BOTH normalization variants
(imagenet, beit05) sequentially: stage 1 (LP, frozen) → stage 2 (FT + EMA) →
inline USFM-aware DARK eval (scored @S=518). Mirrors 00_train_usfm_baseline.ipynb
but headless so it survives a long unattended run (nohup/background).

Usage:
  python _run_usfm_both.py --mode full           # both variants, 5+35 ep each
  python _run_usfm_both.py --mode smoke           # both variants, 1+1 ep (sanity)
  python _run_usfm_both.py --mode full --variants imagenet   # one variant
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
while not (REPO / "pyproject.toml").exists():
    if REPO == REPO.parent:
        raise RuntimeError("repo root not found (expected a pyproject.toml above this file)")
    REPO = REPO.parent
EXP_DIR = REPO / "experiments" / "07_usfm"
for _p in (str(REPO / "experiments" / "_shared_src"), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Engine uses DATA_ROOT_PATH="data" and split CSVs "experiments/SPLITS/..."
# relative to the REPO ROOT (the notebook runs from there via nb_bootstrap).
# _run_usfm.sh cd's into experiments/07_usfm, so we MUST chdir back to REPO
# or KeypointDataset looks for data/csv under 07_usfm and dies.
import os as _os
_os.chdir(REPO)

PREPROC = {
    "imagenet": "clahe_clip_sharpen_speckle",
    "beit05": "clahe_clip_sharpen_speckle_beit05",
}
TRAIN_CSV = "experiments/SPLITS/train.csv"
VAL_CSV = "experiments/SPLITS/internal_val.csv"


class _Tee(io.TextIOBase):
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s: st.write(x); st.flush()
        return len(x)
    def flush(self):
        for st in self.s: st.flush()


def _run_stage(train_mod, log_path, epochs, **kwargs):
    train_mod.NUM_EPOCHS = epochs
    logf = open(log_path, "w", buffering=1)
    o, e = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(o, logf), _Tee(e, logf)
    t0 = time.time()
    try:
        res = train_mod.main(**kwargs)
    finally:
        sys.stdout, sys.stderr = o, e
        logf.flush(); logf.close()
    res["wall_time_sec"] = round(time.time() - t0, 1)
    return res


def run_variant(variant: str, mode: str):
    import nb_usfm_train as train_mod
    import importlib
    importlib.reload(train_mod)  # reset NUM_EPOCHS between variants
    import _dark_eval_usfm as ev

    preprocess = PREPROC[variant]
    s1_ep, s2_ep = (5, 35) if mode == "full" else (1, 1)
    tag = f"usfm_{variant}"
    s1_name = f"00_{tag}_stage1_heads_only"
    s2_name = f"00_{tag}_stage2_unfreeze"
    s1_dir = EXP_DIR / "run" / s1_name
    s2_dir = EXP_DIR / "run" / s2_name
    s1_ckpt = EXP_DIR / "checkpoints" / (s1_name + ".pth")
    s2_ckpt = EXP_DIR / "checkpoints" / (s2_name + ".pth")
    for d in (s1_dir, s2_dir, EXP_DIR / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)
    for ck in (s1_ckpt, s2_ckpt):
        if ck.exists():
            raise RuntimeError(f"{ck} exists — remove or rename before re-run.")

    print(f"\n{'='*70}\n[USFM] VARIANT={variant}  preprocess={preprocess}  mode={mode} "
          f"(S1 {s1_ep}ep / S2 {s2_ep}ep)\n{'='*70}", flush=True)

    # Stage 1 — frozen encoder, heads only, EMA off
    r1 = _run_stage(train_mod, s1_dir / "train.log", s1_ep,
                    leaky=False, train_split_csv=TRAIN_CSV, val_split_csv=VAL_CSV,
                    model_save_path=str(s1_ckpt), preprocess=preprocess,
                    freeze_encoder=True, head_lr=1e-3, ema_decay=None)
    (s1_dir / "summary.json").write_text(json.dumps(
        {**r1, "run_tag": s1_name, "variant": variant, "preprocess": preprocess, "stage": 1}, indent=2))
    print(f"[USFM] {variant} stage1 best_val_mre={r1.get('best_val_mre')}", flush=True)

    # Stage 2 — unfreeze, gentle LRs, EMA 0.999 fresh
    r2 = _run_stage(train_mod, s2_dir / "train.log", s2_ep,
                    leaky=False, train_split_csv=TRAIN_CSV, val_split_csv=VAL_CSV,
                    model_save_path=str(s2_ckpt), preprocess=preprocess,
                    freeze_encoder=False, init_from_ckpt=str(s1_ckpt),
                    encoder_lr=1e-5, head_lr=5e-5, ema_decay=0.999)
    (s2_dir / "summary.json").write_text(json.dumps(
        {**r2, "run_tag": s2_name, "variant": variant, "preprocess": preprocess, "stage": 2}, indent=2))
    print(f"[USFM] {variant} stage2 best_val_mre={r2.get('best_val_mre')}", flush=True)

    # Inline DARK eval @S=518
    res = ev.evaluate_usfm(ckpt_path=str(s2_ckpt), val_csv=str(REPO / VAL_CSV),
                           data_root=str(REPO / "data"), preprocess=preprocess, score_size=518)
    ev.append_csv(csv_path=str(EXP_DIR / "RESULTS_usfm.csv"), ckpt_name=s2_name,
                  ckpt_notes=f"USFM BEiT-512 LP-FT+EMA supervised; variant={variant}; preprocess={preprocess}",
                  result=res)
    dark = res["dark"]["MEAN"]
    print(f"\n[USFM] {variant} DARK @S518 = {dark:.4f}  (native512 {res['dark_native512']['MEAN']:.4f})  "
          f"vs R07 18.4804 = {dark - 18.4804:+.4f}", flush=True)
    return {"variant": variant, "dark_s518": dark, "native512": res["dark_native512"]["MEAN"],
            "s2_best_val": r2.get("best_val_mre")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--variants", nargs="+", default=["imagenet", "beit05"],
                    choices=["imagenet", "beit05"])
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for v in args.variants:
        results.append(run_variant(v, args.mode))

    print(f"\n{'='*70}\n[USFM] ALL DONE in {(time.time()-t0)/60:.1f} min\n{'='*70}")
    print(f"{'variant':12} {'DARK@S518':>10} {'native512':>10} {'vs R07':>8}")
    for r in results:
        print(f"{r['variant']:12} {r['dark_s518']:>10.4f} {r['native512']:>10.4f} "
              f"{r['dark_s518']-18.4804:>+8.4f}")
    print("R07 DINOv2-S leader = 18.4804 DARK @S518")
