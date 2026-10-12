"""_dark_eval_usfm.py — DARK/argmax decoder eval for USFM checkpoints.

Mirror of src/eval/decode_compare.evaluate_ckpt_both_decoders, but builds
the **USFM BEiT-512** model (via the shared usfm_encoder factory) instead
of the hardcoded DINOv2-S@518 in decode_compare._build_model. Using the
shared factory guarantees train↔eval architecture parity (reviewer F9).

MRE SCORING SPACE (reviewer F13 — the 512-vs-518 trap):
  MRE = ||pred_norm - gt_norm|| * S, where the normalized error is
  scale-free (independent of model input size) and S is just the pixel
  yardstick. The folder-01 leader gate (18.4804 DARK) is reported at
  S = 518. USFM runs at input 512. To compare apples-to-apples we score
  the normalized error at the SAME S = 518 (`score_size`, the gate
  reference), and ALSO print the native-512 number for transparency.
  Scoring 512-input predictions at S=518 is legitimate: the model
  predicts in [0,1] normalized coords either way; S only rescales the
  display unit. (True challenge metric is per-image original-size px,
  produced separately by the submission pipeline.)

Run:
  python _dark_eval_usfm.py --ckpt <path> --preprocess <name> [--score-size 518]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE
while not (_REPO / "pyproject.toml").exists():
    _REPO = _REPO.parent
for _p in (
    str(_REPO / "experiments" / "_shared_src"),
    str(_REPO / "Foundation-Model-for-Ultrasound-Biometry" / "baseline"),
    str(_REPO / "experiments" / "07_usfm" / "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INPUT_SIZE = 512  # default / nb00 native; nb01 passes --input-size {256,384}
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
SCORE_SIZE_DEFAULT = 518  # folder-01 leader yardstick (gate 18.4804 DARK)


def _build_task_configs(dataframe):
    EXTRA = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
    configs, seen = [], set()
    for _, row in dataframe.iterrows():
        task_name = str(row["task_name"]); task_id = str(row["task_id"])
        if task_name != "Regression" and task_id not in EXTRA:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append({"task_id": task_id, "task_name": "Regression",
                        "num_classes": int(row["num_classes"])})
    if not configs:
        raise ValueError("No keypoint tasks found.")
    return configs


def _build_val_loader(data_root, val_csv, preprocess, batch_size=8, input_size=INPUT_SIZE):
    from dataset import KeypointDataset
    from utils import keypoint_collate_fn
    from src.training.albu_transforms import build_transforms

    val_t = build_transforms(preprocess=preprocess, input_size=input_size, training=False)
    full = KeypointDataset(data_root=data_root, transforms=val_t,
                           heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA)
    task_configs = _build_task_configs(full.dataframe)
    val_paths = pd.read_csv(val_csv)["image_path"].astype(str).tolist()
    p2i = {str(p): i for i, p in enumerate(full.dataframe["image_path"].astype(str))}
    val_idx = [p2i[p] for p in val_paths if p in p2i]
    if not val_idx:
        raise RuntimeError(f"No val samples resolved from {val_csv}")
    vds = KeypointDataset(data_root=data_root, transforms=val_t,
                          heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA)
    vds.dataframe = full.dataframe.reset_index(drop=True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(vds, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True, collate_fn=keypoint_collate_fn)
    return loader, task_configs


def _build_usfm_model(task_configs, device, ckpt_path, input_size=INPUT_SIZE):
    from model_factory import MultiTaskModelFactory
    from usfm_encoder import attach_usfm_encoder
    model = MultiTaskModelFactory(
        encoder_name="beit_base_patch16_224", encoder_weights=None,
        task_configs=task_configs, heatmap_size=HEATMAP_SIZE).to(device)
    # arch-only swap (weights come from the ckpt below). img_size MUST match
    # the trained ckpt's rel_pos_bias window (grid = input_size//16) or the
    # strict state_dict load below shape-mismatches.
    attach_usfm_encoder(model, img_size=input_size, weights_path=None,
                        device=device, load_weights=False, verbose=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # [RELATION-AWARE LOAD] nb04 ckpts carry extra `relation.*` keys — the M1 R4b
    # relation GCN is TRAIN-LOSS-ONLY (no inference role), so those keys are absent
    # from the base decode model. Load non-strict and assert the ONLY unexpected
    # keys are relation.* with NO missing keys. For relation-free ckpts (nb00/nb02/
    # nb03) unexpected+missing are both empty -> IDENTICAL to the old strict load,
    # so this preserves the nb03 A/B decode byte-for-byte.
    _res = model.load_state_dict(state, strict=False)
    _missing = list(_res.missing_keys)
    _unexpected = [k for k in _res.unexpected_keys if not k.startswith("relation.")]
    if _missing or _unexpected:
        raise RuntimeError(
            f"[eval load ABORT] key mismatch beyond {{relation.*}}.\n"
            f"  missing: {_missing[:8]}\n  unexpected(non-relation): {_unexpected[:8]}"
        )
    model.eval()
    return model


def _mre_batch(pred_norm, gt_norm, score_size):
    p = pred_norm.detach().cpu().numpy().copy()
    g = gt_norm.detach().cpu().numpy().copy()
    p[:, 0::2] *= score_size; p[:, 1::2] *= score_size
    g[:, 0::2] *= score_size; g[:, 1::2] *= score_size
    pp = p.reshape(p.shape[0], -1, 2); gg = g.reshape(g.shape[0], -1, 2)
    return float(np.mean(np.sqrt(((pp - gg) ** 2).sum(-1))))


def evaluate_usfm(ckpt_path, val_csv, data_root="data", preprocess="imagenet_default",
                  score_size=SCORE_SIZE_DEFAULT, device=None, input_size=INPUT_SIZE):
    from src.eval.dark_decoder import decode_heatmaps_argmax, decode_heatmaps_dark
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[USFM eval {Path(ckpt_path).name}] dev={dev} preprocess={preprocess} "
          f"score_size={score_size} (native input {input_size}, grid {input_size//16})")

    loader, task_configs = _build_val_loader(data_root, val_csv, preprocess, input_size=input_size)
    print(f"  val samples: {len(loader.dataset)}")
    model = _build_usfm_model(task_configs, dev, ckpt_path, input_size=input_size)

    mre_arg, mre_dark = defaultdict(list), defaultdict(list)
    mre_arg512, mre_dark512 = defaultdict(list), defaultdict(list)
    n_total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[USFM infer+decode]"):
            images = batch["image"].to(dev)
            labels = batch["label"]; task_ids = batch["task_id"]
            n_total += len(task_ids)
            for tid in sorted(set(task_ids)):
                idx = [i for i, t in enumerate(task_ids) if t == tid]
                imgs = images[idx]
                lbls = torch.stack([labels[i] for i in idx], 0).to(dev)
                hm = torch.sigmoid(model(imgs, task_id=tid))
                pa = decode_heatmaps_argmax(hm)
                pd_ = decode_heatmaps_dark(hm, smooth_sigma=HEATMAP_SIGMA)
                mre_arg[tid].append(_mre_batch(pa, lbls, score_size))
                mre_dark[tid].append(_mre_batch(pd_, lbls, score_size))
                mre_arg512[tid].append(_mre_batch(pa, lbls, input_size))
                mre_dark512[tid].append(_mre_batch(pd_, lbls, input_size))
    wall = time.time() - t0

    def _agg(d):
        pt = {tid: float(np.mean(v)) for tid, v in d.items() if v}
        pt["MEAN"] = float(np.mean(list(pt.values()))) if pt else float("nan")
        return pt

    return {
        "argmax": _agg(mre_arg), "dark": _agg(mre_dark),
        # key name kept as *_native512 for back-compat with _run_usfm_both.py
        # (nb00 runner); the value is native-INPUT scoring (input_size, may be 256).
        "argmax_native512": _agg(mre_arg512), "dark_native512": _agg(mre_dark512),
        "preprocess": preprocess, "score_size": score_size,
        "input_size": input_size, "n_val_samples": n_total, "wall_sec": round(wall, 1),
    }


def append_csv(csv_path, ckpt_name, ckpt_notes, result,
               task_order=("A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur")):
    cols = (["ckpt", "decoder", "score_size", "input_size", "n_val_samples",
             "decode_wall_sec", "val_mre_mean"]
            + [f"MRE_{t}" for t in task_order] + ["ckpt_notes"])

    def _row(dec, pt):
        r = {"ckpt": ckpt_name, "decoder": dec, "score_size": result["score_size"],
             "input_size": result["input_size"], "n_val_samples": result["n_val_samples"],
             "decode_wall_sec": result["wall_sec"],
             "val_mre_mean": round(pt.get("MEAN", float("nan")), 4)}
        for t in task_order:
            v = pt.get(t, float("nan"))
            r[f"MRE_{t}"] = round(v, 4) if v == v else "nan"
        r["ckpt_notes"] = ckpt_notes
        return r

    rows = [_row("argmax", result["argmax"]), _row("dark", result["dark"])]
    csv_path = Path(csv_path)
    df_new = pd.DataFrame(rows, columns=cols)
    df_out = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True) if csv_path.exists() else df_new
    df_out.to_csv(csv_path, index=False)
    print(f"  → wrote 2 rows to {csv_path} (now {len(df_out)} total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-csv", default=str(_REPO / "experiments" / "SPLITS" / "internal_val.csv"))
    ap.add_argument("--data-root", default=str(_REPO / "data"))
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--score-size", type=int, default=SCORE_SIZE_DEFAULT)
    ap.add_argument("--input-size", type=int, default=INPUT_SIZE)  # nb01: 256/384
    ap.add_argument("--csv", default=str(_HERE / "RESULTS_usfm.csv"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    res = evaluate_usfm(args.ckpt, args.val_csv, args.data_root,
                        args.preprocess, args.score_size, input_size=args.input_size)
    print("\n=== USFM DECODER COMPARISON ===")
    print(f"argmax MRE mean @S={args.score_size}: {res['argmax']['MEAN']:.4f} px "
          f"(native{res['input_size']}: {res['argmax_native512']['MEAN']:.4f})")
    print(f"DARK   MRE mean @S={args.score_size}: {res['dark']['MEAN']:.4f} px "
          f"(native{res['input_size']}: {res['dark_native512']['MEAN']:.4f})")
    print(f"\nfolder-01 leader R07 (DINOv2-S @518 DARK): 18.4804 px")
    print(f"USFM vs R07 (DARK @S=518): {res['dark']['MEAN'] - 18.4804:+.4f} px")
    append_csv(args.csv, Path(args.ckpt).stem, args.notes, res)
