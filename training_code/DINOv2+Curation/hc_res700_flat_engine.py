"""hc_res700_flat_engine.py — the 24.28 LB CHAMPION recipe on FLAT cardiac data +
ONE variable: the HC labeled supervised leg runs at input_size=700 instead of 518.

WHAT THIS IS (and how it differs from hc_res700_engine.py)
---------------------------------------------------------
`hc_res700_engine.py` (the PADDED sibling in this folder) trains on enrich_v6, whose
cardiac (A4C/IVC/PLAX/PSAX) rows are the `images_splitfixed` de-split PADDED panels
(crop-to-one-panel + black bars, coords shifted 190-243px). The organizer TEST is
FLAT (val_data fills ~95% of the frame; only 2/619 are dual-pane), so that engine
carries a train(padded)->test(flat) mismatch — the failure mode that sank the sub15
submission (LB 27.81, IVC +25.91).

THIS engine removes that confound: it mirrors the ESTABLISHED 24.28 LB deliverable
EXACTLY (nb04 conf-only UniMatch, 2-stage LP-FT + EMA(0.999) + per-task conf gate,
COLD-START stock DINOv2, cardiac loaded FLAT) and changes exactly ONE thing: HC's
labeled supervised resolution 518 -> 700. So the A/B isolates HC@700 against the
running LB leader (24.28), on the SAME flat-cardiac footing as the deployment test.

PROVENANCE OF THE 24.28 CHAMPION (authoritative: alfonso_7/results/config_snapshot.yaml)
--------------------------------------------------------------------------------------
  * train/val CSV = experiments/SPLITS/{train,internal_val}.csv (leaky=false), the
    audited 5407/1361 split (split_metadata.json build 2026-05-19). Cardiac paths are
    FLAT `<TASK>/DCM_*.png` | `<TASK>/PNG_*.png` — ZERO images_splitfixed rows.
  * DATA_ROOT = data; unlabeled = data/csv_unlabeled/* + data/images_unlabeled/<TASK>.
  * stage2: enable_unimatch, ema 0.999, conf q0.25, conf_max 256, lambda 0.5, ramp 7,
    enc_lr 1e-5, head_lr 5e-5, 35 ep, preprocess clahe_clip_sharpen_speckle.
  * stage1: LP heads-only frozen STOCK encoder, head_lr 1e-3, 5 ep, EMA off, UniMatch off.
  * encoder cold-start = stock timm vit_small_patch14_dinov2.lvd142m weights='pretrained'.

THE FLAT 2-CSV MIRROR (_dataroot_flat)  — why not point at `data` directly
--------------------------------------------------------------------------
`KeypointDataset` GLOBS `<DATA_ROOT>/csv/*.csv` to build its pool. `data/csv/` now
carries de-split/splitfix pollution (A4C_desplit_v5.csv, *_train_splitfix*.csv, ...) +
duplicate-path last-wins hazard, so pointing at `data` would silently swap the champion's
FLAT cardiac for PADDED rows. `_dataroot_flat/` is a deterministic 2-CSV mirror:
  csv/train.csv         = byte copy of SPLITS/train.csv        (sha 4de8...ce30)
  csv/internal_val.csv  = byte copy of SPLITS/internal_val.csv (sha 300d...0ab9)
  images -> data/images                    (FLAT originals, the champion's images)
  split_metadata.json   = byte copy of SPLITS/split_metadata.json (audited shas)
  unlabeled_root/csv_unlabeled/ = the 8 ORIGINAL Apr-22 manifests (A4C/AOP/FA/FUGC/HC/
      IVC/PLAX/PSAX) — VAL_ORG.csv is EXCLUDED (created 2026-06-16, AFTER the champion's
      2026-05-25 training; including it would add unlabeled data the champion never saw =
      a second variable). Cardiac manifests are the FLAT Apr-22 originals, NOT the Jul-07
      splitfixed ones the padded mirror uses.
  unlabeled_root/images_unlabeled -> data/images_unlabeled
=> the glob is {internal_val.csv, train.csv} ONLY; pool = 6768, 0 dups, 0 splitfix rows.

THE SINGLE-VARIABLE CONTRACT
----------------------------
Everything is identical to the 24.28 champion EXCEPT the HC labeled supervised leg,
which moves 518 -> 700:
  * the other 8 tasks stay @518 (labeled + cons)
  * HC's UniMatch cons leg stays @518 (unchanged; only the LABELED leg moves)
  * UniMatch conf-only / EMA(0.999) / conf-gate(q0.25) / LP-FT schedule / preprocess
    clahe_clip_sharpen_speckle / FLAT champion data — all unchanged.
HC is trained ONLY @700 (skipped in the mixed-518 sup batch, exactly like r10 skips
FUGC@518) => the ONE variable is the HC labeled resolution.

BYTE-IDENTICAL HC@700 LEG vs the padded sibling
-----------------------------------------------
The HC@700 mechanism (P1 globals, P2 smoke-read, P3 dedicated-Generator loader + dynamic
backbone swap, P4 sup-skip, P5 highres backward, P6 smoke-cap) is COPIED VERBATIM from
hc_res700_engine.py. preflight() gate G_PATCH imports the padded engine and ASSERTS
_ALL_PATCHES[1:] (P1-P6) are byte-identical, so the ONLY delta between the two engines is
P0 (the DATA_ROOT patch: _dataroot_flat vs _dataroot) + the data-shape gates. RNG isolation
(dedicated torch.Generator + save/restore around the dynamic backbone random init) is
unchanged => the other-8 read stays clean and the ONLY residual cross-task effect is the
shared-encoder gradient from HC@700 supervision (the variable under test, not a confound).

RECIPE (unchanged from the champion)
------------------------------------
  Stage 1 (LP): enc FROZEN (STOCK weights), head_lr 1e-3, 5 ep, EMA off, UniMatch off.
  Stage 2 (FT): unfreeze, enc_lr 1e-5, head_lr 5e-5, 35 ep, EMA 0.999, UniMatch on
                (lambda 0.5, ramp 7), conf gate on (quantile 0.25).
  encoder vit_small_patch14_dinov2.lvd142m, HC leg @700 (dynamic pos_embed), rest @518,
  heatmap 64x64, preprocess clahe_clip_sharpen_speckle.

Run:
  python3 .../hc_res700_flat_engine.py --preflight-only
  python3 .../hc_res700_flat_engine.py --smoke   # 1+1 epoch, capped
  python3 .../hc_res700_flat_engine.py           # full 5+35 (GATED)
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_here = Path(__file__).resolve().parent
REPO = _here
while not (REPO / "pyproject.toml").exists():
    if REPO == REPO.parent:
        raise RuntimeError("repo root not found (need pyproject.toml)")
    REPO = REPO.parent
os.chdir(REPO)

for _p in (REPO / "experiments" / "_shared_src",
           REPO / "Foundation-Model-for-Ultrasound-Biometry" / "baseline",
           REPO / "experiments" / "02_m0_dinov2_SEMISUPERVISED" / "_models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NB04_SRC = (REPO / "experiments" / "02_m0_dinov2_SEMISUPERVISED" / "_models" /
            "nb04_train_unimatch_conf_only.py")
# DATA_ROOT = the FLAT 2-CSV mirror (NOT SPLITS/enrich_v6 padded, NOT bare `data` whose
# csv/ carries splitfix pollution). csv/ globs to EXACTLY {internal_val.csv, train.csv}.
FLAT = _here / "_dataroot_flat"
TRAIN_CSV = FLAT / "csv" / "train.csv"
VAL_CSV = FLAT / "csv" / "internal_val.csv"
UNLABELED_ROOT = FLAT / "unlabeled_root"
META = FLAT / "split_metadata.json"
PREPROCESS = "clahe_clip_sharpen_speckle"

HC_INPUT_SIZE = 700  # THE ONE VARIABLE. None/518 => byte-identical champion (control).

EXTRA = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}

# ---- the champion DATA patch (P0) ----  [ONLY line that differs from the padded engine]
_PATCH_DATA = ('DATA_ROOT_PATH = "data"', f'DATA_ROOT_PATH = {str(FLAT)!r}')

# ---- P1: module globals (os import + HC_INPUT_SIZE default-None no-op) ----
_PATCH_GLOBALS = (
    "INPUT_SIZE = 518",
    "INPUT_SIZE = 518\n"
    "import os as _os  # [HC-RES700]\n"
    "HC_INPUT_SIZE = None  # [HC-RES700] None=>byte-identical champion; 700=>HC labeled leg @700",
)

# ---- P2: read the smoke step-cap env at main entry (default 0 = no cap = full) ----
_PATCH_SMOKE_READ = (
    "    set_seed(RANDOM_SEED)",
    "    _HC_SMOKE_STEPS = int(_os.environ.get(\"HC_SMOKE_STEPS\", \"0\") or 0)  # [HC-RES700]\n"
    "    if _HC_SMOKE_STEPS:\n"
    "        print(f\"[HC-RES700 SMOKE] HC_SMOKE_STEPS={_HC_SMOKE_STEPS} -> cap train steps/epoch\")\n"
    "    set_seed(RANDOM_SEED)",
)

# ---- P3: HC-700 labeled loader (dedicated Generator) + dynamic backbone swap ----
_PATCH_HC_LOADER = (
    """    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
    ).to(device)
""",
    """    # [HC-RES700 LABELED LEG BEGIN] dedicated HC-only highres labeled loader.
    # When HC_INPUT_SIZE is set, HC sup loss comes from THIS loader (source-resized
    # at HC_INPUT_SIZE, SAME plain KeypointDataset + preprocess as the champion labeled
    # leg) and HC is SKIPPED in the mixed-518 sup batch => the single variable is HC's
    # labeled resolution. RNG isolation: a DEDICATED torch.Generator drives shuffle +
    # worker seeding so the GLOBAL rng stream (other-8 aug draws) stays byte-identical.
    # No-op (loaders stay None; no swap) when HC_INPUT_SIZE is None => byte-identical.
    hc_lab_loader = None
    hc_lab_iter = None
    if HC_INPUT_SIZE is not None:
        _hc_res = int(HC_INPUT_SIZE)
        assert _hc_res % 14 == 0, f"[HC-RES700] HC_INPUT_SIZE={_hc_res} must be a multiple of 14 (patch14)"
        _hc_tf = build_transforms(preprocess=preprocess, input_size=_hc_res, training=True)
        _hc_ds = KeypointDataset(
            data_root=DATA_ROOT_PATH, transforms=_hc_tf,
            heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA,
        )
        _hc_ds.dataframe = temp_dataset.dataframe.reset_index(drop=True)
        _hc_tid = _hc_ds.dataframe["task_id"].astype(str)
        _hc_train_idx = [i for i in train_indices if _hc_tid.iloc[i] == "HC"]
        if not _hc_train_idx:
            raise ValueError("[HC-RES700] HC_INPUT_SIZE set but no HC train samples found.")
        _hc_gen = torch.Generator()
        _hc_gen.manual_seed(RANDOM_SEED)
        hc_lab_subset = torch.utils.data.Subset(_hc_ds, _hc_train_idx)
        hc_lab_loader = torch.utils.data.DataLoader(
            hc_lab_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
            pin_memory=True, collate_fn=keypoint_collate_fn, drop_last=True,
            generator=_hc_gen,
        )
        hc_lab_iter = iter(hc_lab_loader)
        print(f"[HC-RES700] HC labeled highres loader: {len(_hc_train_idx)} HC train "
              f"samples at input_size={_hc_res}; HC SKIPPED in the mixed-518 sup batch. "
              f"(dedicated Generator seed={RANDOM_SEED} -> global RNG stream preserved)")
    # [HC-RES700 LABELED LEG END]

    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
    ).to(device)

    # [HC-RES700 DYNAMIC BACKBONE BEGIN] swap the shared DINOv2 backbone to
    # dynamic_img_size so it accepts HC's higher input_size (pos_embed bicubic-resampled
    # per grid; CLS prefix excluded). VERIFIED bit-identical to the static backbone at
    # 518 (max|d|=0.0, 0 missing/unexpected keys), so the other 8 tasks + HC@518-eval
    # stay byte-identical. State-dict keys are unchanged, so the stage2 warm-start load
    # is unaffected. The create_model random init is wrapped in a CPU+CUDA RNG
    # save/restore -> ZERO net global RNG consumption -> head init + downstream draws
    # match the champion.
    if HC_INPUT_SIZE is not None:
        import timm as _timm
        _rng_cpu = torch.get_rng_state()
        _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _dyn_bb = _timm.create_model(ENCODER, pretrained=False, num_classes=0,
                                     dynamic_img_size=True)
        torch.set_rng_state(_rng_cpu)
        if _rng_cuda is not None:
            torch.cuda.set_rng_state_all(_rng_cuda)
        _dyn_bb.load_state_dict(model.encoder.backbone.state_dict())
        model.encoder.backbone = _dyn_bb.to(device)
        print(f"[HC-RES700] dynamic_img_size backbone swapped in — HC leg runs at "
              f"input_size={int(HC_INPUT_SIZE)} (other tasks stay {INPUT_SIZE}); "
              f"pos_embed resampled per grid; RNG state restored (net-zero).")
    # [HC-RES700 DYNAMIC BACKBONE END]
""",
)

# ---- P4: skip HC in the mixed-518 sup batch (HC served by the 700 loader) ----
_PATCH_SUP_SKIP = (
    """            for current_task_id in sorted(set(task_ids)):
                task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]""",
    """            for current_task_id in sorted(set(task_ids)):
                # [HC-RES700] HC sup is served by the dedicated 700 loader -> skip @518.
                if HC_INPUT_SIZE is not None and current_task_id == "HC":
                    continue
                task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]""",
)

# ---- P5: HC-700 supervised backward (grad-accumulated into the same outer batch) ----
_PATCH_HC_BACKWARD = (
    """            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — cons forward(s) +
            # backward(s) [grad-accumulating, λ-weighted].
            cons_batch_value = 0.0""",
    """            # [HC-RES700 SUP BACKWARD BEGIN] HC highres supervised backward, drawn
            # from the dedicated 700 loader, grad-accumulated into the SAME outer batch
            # (mirrors r10's fugc_lab leg). Simple sup path: forward -> sigmoid -> MSE to
            # the precomputed 64x64 heatmap target. No-op when hc_lab_loader is None.
            if hc_lab_loader is not None:
                try:
                    _hl = next(hc_lab_iter)
                except StopIteration:
                    hc_lab_iter = iter(hc_lab_loader)
                    _hl = next(hc_lab_iter)
                _hl_imgs = _hl["image"].to(device)
                _hl_hm = torch.stack(_hl["heatmap"], 0).to(device)
                _hl_logits = model(_hl_imgs, task_id="HC")
                _hl_loss = F.mse_loss(torch.sigmoid(_hl_logits), _hl_hm)
                _hl_loss.backward()
                epoch_train_losses["HC"].append(float(_hl_loss.item()))
            # [HC-RES700 SUP BACKWARD END]

            # [SSL PATCH UNIMATCH-NB00 BEGIN refactor] — cons forward(s) +
            # backward(s) [grad-accumulating, λ-weighted].
            cons_batch_value = 0.0""",
)

# ---- P6: step cap for the smoke (no-op when _HC_SMOKE_STEPS==0) ----
_PATCH_SMOKE_LOOP = (
    """        for batch in loop:
            images = batch["image"].to(device)
            task_ids = batch["task_id"]""",
    """        for _hc_bi, batch in enumerate(loop):
            if _HC_SMOKE_STEPS and _hc_bi >= _HC_SMOKE_STEPS:  # [HC-RES700 SMOKE cap]
                break
            images = batch["image"].to(device)
            task_ids = batch["task_id"]""",
)

_ALL_PATCHES = [_PATCH_DATA, _PATCH_GLOBALS, _PATCH_SMOKE_READ, _PATCH_HC_LOADER,
                _PATCH_SUP_SKIP, _PATCH_HC_BACKWARD, _PATCH_SMOKE_LOOP]

# Expected FLAT champion split shape (from SPLITS/split_metadata.json, build 2026-05-19).
EXP_POOL = 6768      # train 5407 + internal_val 1361 (2-CSV mirror glob)
EXP_TRAIN = 5407
EXP_VAL = 1361
EXP_SPLITFIX = 0     # FLAT gate: cardiac is flat => ZERO splitfix/padded rows in the pool
EXP_HC_TRAIN = 799   # HC rows in the train split (sizes the 700 leg; non-empty gate)

# markers that would betray a PADDED / de-split cardiac row (must be ABSENT for FLAT)
_SPLITFIX_MARKERS = ("images_splitfixed", "_valfix", "_bmode", "splitfix", "desplit", "padded")


def _sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(rel_path: str, root: Path):
    """Mirror KeypointDataset._resolve_image_path (dataset.py:62-77) EXACTLY."""
    rel_norm = os.path.normpath(rel_path)
    cleaned = rel_norm
    while cleaned.startswith(".." + os.sep):
        cleaned = cleaned[3:]
    for r in (os.path.join(str(root), "images"), str(root)):
        direct = os.path.normpath(os.path.join(r, cleaned))
        if os.path.isfile(direct):
            return direct
    return None


def preflight():
    """Hard gates G0-G7. Abort BEFORE the GPU if anything does not add up."""
    print("=" * 82)
    print("PRE-FLIGHT — 24.28 champion recipe on FLAT cardiac + HC labeled leg @700 (single variable)")
    print("=" * 82)

    # G0: engine source still has EVERY patch anchor and is UNMODIFIED by us.
    src = NB04_SRC.read_text()
    for i, (anchor, _) in enumerate(_ALL_PATCHES):
        assert anchor in src, f"G0 FAIL: patch anchor P{i} missing (nb04 source changed?):\n{anchor[:80]!r}"
        assert src.count(anchor) == 1, f"G0 FAIL: patch anchor P{i} not unique ({src.count(anchor)}x)"
    print(f"  G0 all {len(_ALL_PATCHES)} patch anchors found & unique      OK  ({NB04_SRC.name}, read-only)")

    # G_PATCH: HC@700 leg (P1-P6) is BYTE-IDENTICAL to the padded sibling => the ONLY
    # delta between the two engines is P0 (DATA_ROOT) + the data-shape gates below.
    import hc_res700_engine as _PAD
    assert _ALL_PATCHES[1:] == _PAD._ALL_PATCHES[1:], (
        "G_PATCH FAIL: P1-P6 differ from hc_res700_engine.py -> HC@700 leg is NOT the "
        "single variable")
    assert _ALL_PATCHES[0] != _PAD._ALL_PATCHES[0], \
        "G_PATCH FAIL: P0 DATA patch equals the padded engine (mirror not repointed?)"
    assert str(FLAT) in _ALL_PATCHES[0][1] and str(_PAD.V6) in _PAD._ALL_PATCHES[0][1]
    print(f"  G_PATCH P1-P6 byte-identical to hc_res700_engine.py; ONLY P0 differs "
          f"(_dataroot_flat vs _dataroot)  OK")

    # G1: the POOL the loader will actually build = the 2-CSV FLAT mirror (no pollution).
    files = sorted(glob.glob(os.path.join(str(FLAT), "csv", "*.csv")))
    assert [Path(f).name for f in files] == ["internal_val.csv", "train.csv"], \
        f"G1 FAIL: unexpected CSVs in _dataroot_flat/csv: {[Path(f).name for f in files]}"
    pool = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).reset_index(drop=True)
    pool = pool[pool["task_name"].astype(str).eq("Regression")
                | pool["task_id"].astype(str).isin(EXTRA)].reset_index(drop=True)
    n_dup = len(pool) - pool["image_path"].nunique()
    assert len(pool) == EXP_POOL, f"G1 FAIL: pool={len(pool)} expected {EXP_POOL}"
    assert n_dup == 0, f"G1 FAIL: {n_dup} duplicate paths in pool -> last-wins label hazard"
    print(f"  G1 pool rows = {len(pool)}, unique = {pool['image_path'].nunique()}, "
          f"duplicates = {n_dup}  -> DETERMINISTIC labels (2-CSV FLAT mirror)  OK")

    # G2: split sizes + zero overlap in POOL-INDEX space + HC non-empty.
    p2i = {str(p): i for i, p in enumerate(pool["image_path"].astype(str))}
    tr_df = pd.read_csv(TRAIN_CSV)
    tr = tr_df["image_path"].astype(str).tolist()
    va = pd.read_csv(VAL_CSV)["image_path"].astype(str).tolist()
    assert len(tr) == EXP_TRAIN, f"G2 FAIL: train={len(tr)} expected {EXP_TRAIN}"
    assert len(va) == EXP_VAL, f"G2 FAIL: val={len(va)} expected {EXP_VAL}"
    miss = [p for p in tr + va if p not in p2i]
    assert not miss, f"G2 FAIL: {len(miss)} split paths unresolved in pool, e.g. {miss[:3]}"
    ti, vi = {p2i[p] for p in tr}, {p2i[p] for p in va}
    assert not (ti & vi), f"G2 FAIL: {len(ti & vi)} pool indices in BOTH train and val"
    n_hc_tr = int((tr_df["task_id"].astype(str) == "HC").sum())
    assert n_hc_tr == EXP_HC_TRAIN, f"G2 FAIL: HC train rows={n_hc_tr} expected {EXP_HC_TRAIN}"
    print(f"  G2 train {len(tr)} / val {len(va)} / overlap {len(ti & vi)}, "
          f"HC train rows {n_hc_tr} (700 leg non-empty)  OK")

    # G3: FLAT gate — ZERO padded/splitfix/de-split cardiac rows in the pool (proves the
    # cardiac swapped out by the padded engine is NOT present here).
    _bad = pool["image_path"].astype(str).str.contains(
        "|".join(_SPLITFIX_MARKERS), case=False, regex=True)
    n_splitfix = int(_bad.sum())
    assert n_splitfix == EXP_SPLITFIX, (
        f"G3 FAIL: {n_splitfix} splitfix/padded rows in pool (expected {EXP_SPLITFIX}); "
        f"e.g. {pool.loc[_bad, 'image_path'].head(3).tolist()}")
    n_card = int(pool["task_id"].astype(str).isin({"A4C", "IVC", "PLAX", "PSAX"}).sum())
    print(f"  G3 FLAT gate: {n_splitfix} splitfix/padded rows in pool (of {n_card} cardiac) "
          f"-> cardiac is FLAT  OK")

    # G4: EVERY pool image physically resolves under the FLAT mirror.
    unres = [p for p in pool["image_path"].astype(str) if _resolve(p, FLAT) is None]
    assert not unres, f"G4 FAIL: {len(unres)} images unresolvable under {FLAT}, e.g. {unres[:3]}"
    print(f"  G4 all {len(pool)} pool images resolve under _dataroot_flat mirror  OK")

    # G5: sha256 of the (copied) CSVs == split_metadata audited build (2026-05-19).
    meta = json.loads(META.read_text())
    exp_sha = meta["sha256"]
    got_tr, got_va = _sha256(TRAIN_CSV), _sha256(VAL_CSV)
    assert got_tr == exp_sha["train.csv"], (
        f"G5 FAIL: train.csv sha256 {got_tr[:12]} != audited {exp_sha['train.csv'][:12]}")
    assert got_va == exp_sha["internal_val.csv"], (
        f"G5 FAIL: internal_val.csv sha256 {got_va[:12]} != audited {exp_sha['internal_val.csv'][:12]}")
    print(f"  G5 CSV sha256 == split_metadata audited (FLAT champion split)  OK")

    # G6: unlabeled corpus resolves and is non-empty; VAL_ORG.csv EXCLUDED (champion parity).
    assert UNLABELED_ROOT.is_dir(), f"G6 FAIL: unlabeled root missing: {UNLABELED_ROOT}"
    man = sorted((UNLABELED_ROOT / "csv_unlabeled").glob("*.csv"))
    assert man, f"G6 FAIL: no manifests in {UNLABELED_ROOT / 'csv_unlabeled'}"
    man_names = [m.name for m in man]
    assert "VAL_ORG.csv" not in man_names, (
        "G6 FAIL: VAL_ORG.csv present -> would add post-champion unlabeled data (2nd variable)")
    assert (UNLABELED_ROOT / "images_unlabeled").is_dir(), "G6 FAIL: images_unlabeled missing"
    n_unl = sum(len(pd.read_csv(m)) for m in man)
    assert n_unl > 0, "G6 FAIL: unlabeled corpus empty"
    print(f"  G6 unlabeled root OK — {len(man)} manifests {man_names}, {n_unl:,} rows "
          f"(VAL_ORG excluded)  OK")

    # G7: cold-start (stock DINOv2) + HC-700 divisibility assert.
    ns_probe = {}
    for line in src.splitlines():
        if line.startswith("ENCODER ") or line.startswith("ENCODER_WEIGHTS "):
            exec(line, ns_probe)  # noqa: S102
    assert ns_probe.get("ENCODER") == "vit_small_patch14_dinov2.lvd142m", \
        f"G7 FAIL: encoder={ns_probe.get('ENCODER')!r} not stock DINOv2"
    assert ns_probe.get("ENCODER_WEIGHTS") == "pretrained", \
        f"G7 FAIL: ENCODER_WEIGHTS={ns_probe.get('ENCODER_WEIGHTS')!r} != 'pretrained'"
    assert HC_INPUT_SIZE is None or int(HC_INPUT_SIZE) % 14 == 0, \
        f"G7 FAIL: HC_INPUT_SIZE={HC_INPUT_SIZE} not divisible by 14 (patch14 grid)"
    _grid = None if HC_INPUT_SIZE is None else int(HC_INPUT_SIZE) // 14
    print(f"  G7 encoder = {ns_probe['ENCODER']} weights='pretrained' (STOCK timm) cold-start  OK")
    print(f"     HC_INPUT_SIZE = {HC_INPUT_SIZE} -> patch grid {_grid}x{_grid} (700/14=50)  OK")

    print("=" * 82)
    print("PRE-FLIGHT PASSED — safe to burn GPU\n")
    return dict(pool_rows=len(pool), pool_dups=n_dup, train=len(tr), val=len(va),
                overlap=len(ti & vi), splitfix_rows=n_splitfix, unlabeled_rows=int(n_unl),
                hc_train_rows=n_hc_tr, hc_input_size=HC_INPUT_SIZE,
                train_sha=got_tr, val_sha=got_va)


def _build_patched_module():
    src = NB04_SRC.read_text()
    for i, (anchor, repl) in enumerate(_ALL_PATCHES):
        if anchor not in src:
            raise RuntimeError(f"patch anchor P{i} not found (nb04 source changed?): {anchor[:80]!r}")
        src = src.replace(anchor, repl, 1)
    ns = {"__name__": "nb04_hc_res700_flat_patched", "__file__": str(NB04_SRC)}
    exec(compile(src, str(NB04_SRC), "exec"), ns)  # noqa: S102 (real engine + surgical patches)
    assert ns["DATA_ROOT_PATH"] == str(FLAT), "P0 data patch did not take"
    assert "HC_INPUT_SIZE" in ns, "P1 global patch did not take"
    return ns


def run(tag: str, s1_epochs: int, s2_epochs: int, hc_input_size=HC_INPUT_SIZE,
        smoke_steps: int = 0):
    out = _here / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    gates = preflight()

    if smoke_steps and smoke_steps > 0:
        os.environ["HC_SMOKE_STEPS"] = str(int(smoke_steps))
    else:
        os.environ.pop("HC_SMOKE_STEPS", None)

    s1_ckpt = out / "stage1_heads_only.pth"
    s2_ckpt = out / "stage2_unfreeze.pth"
    ns = _build_patched_module()
    main = ns["main"]

    # ===== STAGE 1 — LP: encoder FROZEN at STOCK weights, heads only. HC leg @700. =====
    ns["NUM_EPOCHS"] = s1_epochs
    ns["HC_INPUT_SIZE"] = hc_input_size
    print(f"\n===== STAGE 1 (LP, heads-only, STOCK frozen encoder) {s1_epochs} ep  "
          f"[HC_INPUT_SIZE={hc_input_size}] =====")
    t0 = time.time()
    r1 = main(leaky=False, train_split_csv=str(TRAIN_CSV), val_split_csv=str(VAL_CSV),
              preprocess=PREPROCESS, freeze_encoder=True, head_lr=1e-3,
              init_from_ckpt=None,
              ema_decay=None, enable_unimatch=False,
              enable_confidence=True, conf_quantile=0.25,
              model_save_path=str(s1_ckpt))
    s1_wall = round(time.time() - t0, 1)
    print(f"[stage1] best_val_mre={r1.get('best_val_mre')}  wall={s1_wall}s")

    # ===== STAGE 2 — FT: unfreeze + UniMatch + EMA, init from OUR stage1. HC leg @700. =====
    ns["NUM_EPOCHS"] = s2_epochs
    ns["HC_INPUT_SIZE"] = hc_input_size
    print(f"\n===== STAGE 2 (FT, unfreeze + UniMatch + EMA) {s2_epochs} ep  "
          f"[HC_INPUT_SIZE={hc_input_size}] =====")
    t0 = time.time()
    r2 = main(leaky=False, train_split_csv=str(TRAIN_CSV), val_split_csv=str(VAL_CSV),
              preprocess=PREPROCESS, freeze_encoder=False, init_from_ckpt=str(s1_ckpt),
              encoder_lr=1e-5, head_lr=5e-5, ema_decay=0.999,
              enable_unimatch=True, lambda_cons_max=0.5, lambda_ramp_epochs=7,
              unlabeled_data_root=str(UNLABELED_ROOT),
              unlabeled_weak_preprocess="ssl_weak_v1",
              unlabeled_strong_preprocess="ssl_strong_v1",
              enable_confidence=True, conf_quantile=0.25,
              model_save_path=str(s2_ckpt))
    s2_wall = round(time.time() - t0, 1)

    summary = dict(
        tag=tag, kind="24.28_champion_recipe_on_FLAT + HC_labeled_leg@700 (single variable)",
        data="FLAT champion split (via _dataroot_flat 2-CSV mirror; SPLITS/{train,internal_val}.csv)",
        variable=f"HC labeled resolution 518 -> {hc_input_size} (other 8 tasks + HC cons stay 518)",
        init="STOCK timm DINOv2 (no FU ckpt) = champion cold-start",
        encoder="vit_small_patch14_dinov2.lvd142m", preprocess=PREPROCESS,
        hc_input_size=hc_input_size, smoke_steps=int(smoke_steps) if smoke_steps else 0,
        s1_epochs=s1_epochs, s2_epochs=s2_epochs,
        stage1_best_val=r1.get("best_val_mre"), stage1_wall_sec=s1_wall,
        stage2_best_val=r2.get("best_val_mre"), stage2_wall_sec=s2_wall,
        stage1_ckpt=str(s1_ckpt), stage2_ckpt=str(s2_ckpt),
        gates=gates,
        note=("nb04 conf-only COLD-START on the FLAT 24.28-champion split with ONE "
              "variable: HC labeled leg @700 (dynamic pos_embed). Cardiac is FLAT (0 "
              "splitfix rows) — the train(padded)->test(flat) confound of the padded "
              "sibling is removed. stage2_best_val is the ARGMAX-518 EMA metric (HC "
              "scored @518 during training); eval_hc_res700_flat.py reproduces it "
              "(provenance) AND scores HC @700 (deployment). LB decides."),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {out / 'summary.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1+1 epoch smoke (step-capped)")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hc-input-size", type=int, default=HC_INPUT_SIZE,
                    help="HC labeled leg input size (700 default; 518 => control).")
    ap.add_argument("--smoke-steps", type=int, default=40,
                    help="train steps/epoch cap for --smoke (0 = full epoch).")
    a = ap.parse_args()
    _hc = None if int(a.hc_input_size) in (0, 518) else int(a.hc_input_size)
    if a.preflight_only:
        preflight()
        sys.exit(0)
    if a.smoke:
        run(a.tag or "00_hc_res700_flat_SMOKE_v1_0722", s1_epochs=1, s2_epochs=1,
            hc_input_size=_hc, smoke_steps=int(a.smoke_steps))
    else:
        run(a.tag or "00_hc_res700_flat_v1_0722", s1_epochs=5, s2_epochs=35,
            hc_input_size=_hc, smoke_steps=0)
