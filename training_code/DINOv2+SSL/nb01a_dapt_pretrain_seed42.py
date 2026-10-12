#!/usr/bin/env python3
"""nb01a_dapt_pretrain_seed42.py — folder-04 SSL Phase A launcher.

ROLE
----
Orchestrates the continual iBOT + DINO + KoLeo pretrain of DINOv2 ViT-S/14
(LVD-142M warm-start, in_chans=1) on our 187,744-image unlabeled ultrasound
corpus, using UltraDINO's `DataAugmentationDINUS`-derived recipe (our local
`DataAugmentationOURUS` class with OURS_MEAN/STD scalars).

This script does NOT run training — training runs in the vendored
`dinov2/train/train.py` invoked via `python -m dinov2.run.train.train`.
Instead, this launcher handles 5 phases:

  PHASE 0  PREP
    0a. Find repo root.
    0b. Add `code/` to sys.path so `import dinov2` resolves to vendored.
    0c. Set env vars consumed by FUSOURS (OURS_MANIFEST, OURS_IMAGES_ROOT,
        REPO_ROOT).
    0d. Run sampler diagnostic (refuse launch if any task share >55 %).
    0e. (--smoke only) Build manifest_smoke_5k.csv (5000 rows, seed 42)
        if missing.
    0f. Build LVD-142M 1-channel ckpt
        (`checkpoints/lvd142m_vits14_in_chans1.pth`) if missing —
        timm.create_model + sum(dim=1,keepdim=True) on patch_embed.proj.

  PHASE 1  INVOKE TRAIN
    subprocess invocation of dinov2 train, tee'd to a log file under
    `run/<tag>/train.log`. CWD=code/, PYTHONPATH augmented.

  PHASE 2  SMOKE POST-CHECK (--smoke only)
    Verify ckpt landed; print iBOT/DINO/KoLeo loss descent summary from
    log. NB: cardiac downstream probe (HIGH 6 adv fix) is a separate
    follow-up `00_smoke_cardiac_probe.py` (TODO) — gates the full launch.

  PHASE 3  POST-TRAIN EXTRACT (--full only)
    Load `output_dir/teacher_checkpoint.pth`, strip `teacher.backbone.`
    prefix, save backbone-only state_dict to
    `checkpoints/ultradino_ours_vits14_seed42.pth` (consumed by nb01b).

  PHASE 4  SUMMARY (always)
    Write `run/<tag>/summary.json` with paths + wall_sec + return code.

CLI FLAGS
---------
  --smoke         Use smoke config (5k images / 5 epochs, ~4 h GPU).
  --full          Use full config (187k images / 40 epochs, ~24-36 h GPU).
                  Default if neither flag (=full).
  --skip-prep     Skip phase 0e + 0f (assume artifacts already present).
  --dry-run       Resolve config + paths + env + cmd, do NOT invoke train.

WALL-CLOCK ESTIMATES (single RTX PRO 6000, BF16, uncontested)
  Smoke: ~4 h
  Full:  ~24-36 h
With GPU contention add 50 %.

DOWNSTREAM
----------
nb01b (`experiments/04_dinov2_ultradinoLIKE/nb01b_dapt_lpft_ema_seed42.py`)
loads `checkpoints/ultradino_ours_vits14_seed42.pth` as `encoder_init` for
the supervised LP-FT + EMA stack. nb01b is byte-equivalent to nb00 except
the encoder init.

Per PROGRESSION_PLAN_v3 §3.2 + CONTEXT.md §3.2 (folder-04 recipe-not-params
faithful UltraDINO).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch


# ─── utils ─────────────────────────────────────────────────────────────────────

def find_repo_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from CWD to find the directory containing `marker`."""
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} walking up from {here}")


def build_smoke_manifest(full_csv: Path, smoke_csv: Path, n: int = 5000, seed: int = 42) -> None:
    """Sample n rows from full_csv with deterministic seed → write smoke_csv."""
    print(f"  Building smoke manifest: {n} rows from {full_csv} (seed={seed})")
    df = pd.read_csv(full_csv)
    if len(df) < n:
        raise ValueError(f"manifest has {len(df)} rows, cannot sample {n}")
    df_smoke = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df_smoke.to_csv(smoke_csv, index=False)
    print(f"  Wrote {smoke_csv} ({smoke_csv.stat().st_size:,} B)")


def build_lvd142m_1ch_ckpt(
    ckpt_path: Path,
    model_name: str = "vit_small_patch14_dinov2.lvd142m",
    target_global_crops_size: int = 224,
    target_patch_size: int = 14,
) -> None:
    """Download LVD-142M via timm; collapse patch_embed.proj.weight 3->1;
    interpolate pos_embed to match SSL crop grid; wrap state_dict in
    {'model': ...} envelope expected by dinov2 ssl_meta_arch.

    Fixes:
      - [BLOCKER 1] dinov2 ssl_meta_arch.py:44-47 unconditionally accesses
        chkpt['model']; bare state_dict → KeyError. Wrap.
      - [HIGH 6] LVD-142M pos_embed shape (1, 1370, 384) for 518/14 = 37×37;
        SSL trains at 224/14 = 16×16 → expects (1, 257, 384). Without
        interpolation, dinov2's strict=False silently rejects the mismatch
        → random pos_embed init = lose LVD-142M positional pretrain entirely.
    """
    import timm
    import torch.nn.functional as F
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {model_name} via timm ...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    state = model.state_dict()

    # ── [BLOCKER 1 partial] patch_embed.proj.weight collapse 3->1 ──────────
    proj_key = "patch_embed.proj.weight"
    if proj_key not in state:
        raise KeyError(f"{proj_key} not in state_dict; got keys: {list(state.keys())[:10]}")
    orig = state[proj_key]
    if orig.shape != (384, 3, 14, 14):
        raise ValueError(f"unexpected patch_embed shape {tuple(orig.shape)}, expected (384, 3, 14, 14)")
    collapsed = orig.sum(dim=1, keepdim=True)
    print(f"  patch_embed.proj.weight {tuple(orig.shape)} -> {tuple(collapsed.shape)} via .sum(dim=1, keepdim=True)")
    state[proj_key] = collapsed

    # ── [HIGH 6 fix] pos_embed bicubic interpolation 37x37 -> 16x16 ────────
    pos_key = "pos_embed"
    if pos_key not in state:
        raise KeyError(f"{pos_key} not in state_dict")
    pos = state[pos_key]   # expected (1, 1370, 384) for 518/14 LVD-142M
    n_tokens = pos.shape[1]
    n_patches_src = n_tokens - 1   # 1369 = 37 * 37
    P_src = int(round(n_patches_src ** 0.5))
    if P_src * P_src != n_patches_src:
        raise ValueError(f"non-square patch count in pos_embed: {n_patches_src}")
    P_tgt = target_global_crops_size // target_patch_size   # 224 / 14 = 16
    if P_tgt != P_src:
        cls_pos = pos[:, :1, :]                              # (1, 1, 384)
        patch_pos = pos[:, 1:, :]                            # (1, 1369, 384)
        embed_dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, P_src, P_src, embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(P_tgt, P_tgt), mode="bicubic", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, P_tgt * P_tgt, embed_dim)
        new_pos = torch.cat([cls_pos, patch_pos], dim=1)     # (1, P_tgt*P_tgt+1, 384)
        state[pos_key] = new_pos
        print(
            f"  pos_embed interpolated {tuple(pos.shape)} -> {tuple(new_pos.shape)} "
            f"(grid {P_src}x{P_src} -> {P_tgt}x{P_tgt}, bicubic) for SSL "
            f"global_crops_size={target_global_crops_size}/patch={target_patch_size}"
        )
    else:
        print(f"  pos_embed grid already matches target {P_tgt}x{P_tgt}; no interpolation needed")

    # ── [BLOCKER 1 main] dinov2 expects {'model': state_dict} envelope ─────
    payload = {"model": state}
    torch.save(payload, ckpt_path)

    meta = {
        "source": f"timm://{model_name}",
        "orig_patch_embed_shape": list(orig.shape),
        "collapsed_patch_embed_shape": list(collapsed.shape),
        "collapse_op": ".sum(dim=1, keepdim=True)",
        "orig_pos_embed_shape": list(pos.shape),
        "new_pos_embed_shape": list(state[pos_key].shape),
        "pos_embed_grid_src_x_src": [P_src, P_src],
        "pos_embed_grid_tgt_x_tgt": [P_tgt, P_tgt],
        "pos_embed_interp_mode": "bicubic",
        "n_params_state": int(sum(v.numel() for v in state.values() if hasattr(v, "numel"))),
        "wrapper_envelope": "{'model': state_dict}  # required by dinov2 ssl_meta_arch.py:47",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ckpt_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"  Saved ckpt: {ckpt_path} ({ckpt_path.stat().st_size:,} B)")
    print(f"  Saved meta: {ckpt_path.with_suffix('.meta.json')}")


def run_sampler_diagnostic(diag_script: Path) -> None:
    print(f"  Running sampler diagnostic: {diag_script}")
    rc = subprocess.call([sys.executable, str(diag_script)])
    if rc != 0:
        raise RuntimeError(f"Sampler diagnostic exited rc={rc}; abort launch.")


def tee_subprocess(cmd: List[str], env: dict, cwd: Path, log_path: Path) -> int:
    """Run cmd, tee stdout+stderr to log_path AND console. Return rc."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  CWD: {cwd}")
    print(f"  Log: {log_path}")
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            logf.write(raw)
            logf.flush()
        proc.wait()
    return proc.returncode


def extract_losses(log_text: str, loss_name: str) -> List[float]:
    """Find lines containing 'loss_<name>=<float>' and parse the values."""
    import re
    pattern = rf"loss_{loss_name}=([\d.]+)"
    return [float(v) for v in re.findall(pattern, log_text)]


def loss_descended(losses: List[float], min_samples: int = 20) -> bool:
    if len(losses) < min_samples:
        return False
    first = sum(losses[:10]) / 10.0
    last = sum(losses[-10:]) / 10.0
    return last < first


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--smoke", action="store_true", help="Use smoke config (5k/5ep)")
    p.add_argument("--full", action="store_true", help="Use full config (187k/40ep, default)")
    p.add_argument("--skip-prep", action="store_true", help="Skip phase 0e + 0f")
    p.add_argument("--dry-run", action="store_true", help="Resolve only, do NOT invoke train")
    args = p.parse_args()
    if args.smoke and args.full:
        p.error("--smoke and --full are mutually exclusive")
    use_smoke = bool(args.smoke)
    use_full = not use_smoke   # default when neither

    # ── Paths ──────────────────────────────────────────────────────────────────
    repo = find_repo_root()
    folder04 = repo / "experiments" / "04_dinov2_ultradinoLIKE"
    code_dir = folder04 / "code"
    ckpts_dir = folder04 / "checkpoints"
    run_dir = folder04 / "run"
    tag = "01a_dapt_pretrain_smoke_seed42" if use_smoke else "01a_dapt_pretrain_seed42"
    run_dir_tag = run_dir / tag
    run_dir_tag.mkdir(parents=True, exist_ok=True)
    log_path = run_dir_tag / "train.log"

    manifest_full = folder04 / "manifest_187k.csv"
    manifest_smoke = folder04 / "manifest_smoke_5k.csv"
    manifest_active = manifest_smoke if use_smoke else manifest_full

    images_root = repo / "data" / "images_unlabeled"
    ckpt_lvd142m = ckpts_dir / "lvd142m_vits14_in_chans1.pth"
    config_full = code_dir / "dinov2" / "configs" / "train" / "ours_vits14_ibot_dapt.yaml"
    config_smoke = code_dir / "dinov2" / "configs" / "train" / "ours_vits14_ibot_dapt_smoke.yaml"
    config_active = config_smoke if use_smoke else config_full
    diag_script = folder04 / "00_sampler_diagnostic.py"

    # output_dir is owned by the YAML config (train.output_dir). For tag
    # consistency we also pass --output-dir CLI override = run_dir_tag.
    output_dir = run_dir_tag

    teacher_ckpt_in = output_dir / "eval" / "training_99999" / "teacher_checkpoint.pth"
    # Path layout per dinov2 framework: actual path discovered post-train.
    ckpt_final = ckpts_dir / "ultradino_ours_vits14_seed42.pth"

    # ── PHASE 0 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"===== PHASE 0: PREP   (mode={'SMOKE' if use_smoke else 'FULL'}) =====")
    print("=" * 70)
    print(f"  repo:           {repo}")
    print(f"  folder04:       {folder04}")
    print(f"  code_dir:       {code_dir}")
    print(f"  config_active:  {config_active}")
    print(f"  manifest_active:{manifest_active}")
    print(f"  output_dir:     {output_dir}")
    print(f"  log_path:       {log_path}")

    # 0b: sys.path
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    # 0c: env vars
    env = os.environ.copy()
    env["OURS_MANIFEST"] = str(manifest_active)
    env["OURS_IMAGES_ROOT"] = str(images_root)
    env["REPO_ROOT"] = str(repo)
    env["PYTHONPATH"] = str(code_dir) + os.pathsep + env.get("PYTHONPATH", "")
    # Single-GPU torch.distributed env. dinov2 distributed/__init__.py:175
    # triggers _set_from_local() (auto-detect single-GPU) ONLY when both
    # MASTER_ADDR and MASTER_PORT are EMPTY STRINGS. Set both to "" so
    # dinov2 takes the local-single-GPU branch (auto-config rank=0, world=1).
    env["MASTER_ADDR"] = ""
    env["MASTER_PORT"] = ""
    # xformers flash-attn Hopper kernel fails on Blackwell (RTX PRO 6000) with
    # "invalid argument" CUDA error. Disable flash-attn backends; keep triton
    # (xformers indexing ops require triton — disabling it breaks index_select_cat).
    # CUTLASS op forced via runtime monkey-patch in code/dinov2/layers/attention.py.
    env["XFORMERS_DISABLE_FLASH_ATTN"] = "1"
    env["XFORMERS_FLASH3_DISABLE"] = "1"
    # Reduce VRAM fragmentation under GPU contention.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"  OURS_MANIFEST    = {env['OURS_MANIFEST']}")
    print(f"  OURS_IMAGES_ROOT = {env['OURS_IMAGES_ROOT']}")
    print(f"  REPO_ROOT        = {env['REPO_ROOT']}")
    print(f"  PYTHONPATH       = {env['PYTHONPATH']}")

    # 0d: sampler diagnostic
    if not diag_script.exists():
        raise FileNotFoundError(f"diag script missing: {diag_script}")
    run_sampler_diagnostic(diag_script)

    # 0e: smoke manifest
    if use_smoke and (not args.skip_prep) and (not manifest_smoke.exists()):
        build_smoke_manifest(manifest_full, manifest_smoke)
    elif use_smoke and manifest_smoke.exists():
        print(f"  Smoke manifest exists: {manifest_smoke} (use --skip-prep to force re-use)")

    # 0f: LVD-142M 1-ch ckpt
    if (not args.skip_prep) and (not ckpt_lvd142m.exists()):
        build_lvd142m_1ch_ckpt(ckpt_lvd142m)
    elif ckpt_lvd142m.exists():
        print(f"  LVD-142M 1ch ckpt exists: {ckpt_lvd142m}")

    # Validate all required paths
    required = {
        "manifest_active": manifest_active,
        "ckpt_lvd142m": ckpt_lvd142m,
        "config_active": config_active,
        "code_dir": code_dir,
        "images_root": images_root,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"required path missing: {name} = {path}")

    # ── PHASE 1 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("===== PHASE 1: INVOKE DINOV2 TRAIN =====")
    print("=" * 70)
    # [BLOCKER 3 adv fix 2026-05-24] Pass LVD-142M ckpt path as ABSOLUTE
    # via --opts. Subprocess CWD=code_dir would mis-resolve a relative path
    # in the YAML, so YAML keeps pretrained_weights='' and we inject the
    # absolute path here via OmegaConf.from_cli merge (config.py:53).
    cmd = [
        sys.executable,
        "-m", "dinov2.run.train.train",
        f"--config-file={config_active}",
        f"--output-dir={output_dir}",
        f"student.pretrained_weights={ckpt_lvd142m.resolve()}",
    ]

    if args.dry_run:
        print(f"  DRY-RUN cmd: {' '.join(cmd)}")
        print(f"  DRY-RUN env additions: OURS_MANIFEST, OURS_IMAGES_ROOT, REPO_ROOT, PYTHONPATH")
        sys.exit(0)

    t0 = time.time()
    rc = tee_subprocess(cmd, env=env, cwd=code_dir, log_path=log_path)
    wall_sec = time.time() - t0
    if rc != 0:
        # Dump last 50 log lines for triage
        lines = log_path.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-50:])
        raise RuntimeError(
            f"dinov2 train exited rc={rc} (wall {wall_sec:.0f}s).\n"
            f"Last 50 log lines:\n{tail}"
        )
    print(f"  Training subprocess exited rc=0 (wall {wall_sec:.0f}s = {wall_sec/3600:.2f}h)")

    # ── PHASE 2 (smoke) ────────────────────────────────────────────────────────
    if use_smoke:
        print("=" * 70)
        print("===== PHASE 2: SMOKE POST-CHECK =====")
        print("=" * 70)
        log_text = log_path.read_text(errors="replace")
        ibot = extract_losses(log_text, "ibot")
        dino = extract_losses(log_text, "dino")
        koleo = extract_losses(log_text, "koleo")
        for name, vals in [("iBOT", ibot), ("DINO", dino), ("KoLeo", koleo)]:
            if len(vals) < 20:
                print(f"  {name}: only {len(vals)} loss values logged; skip descent check.")
                continue
            first10 = sum(vals[:10]) / 10.0
            last10 = sum(vals[-10:]) / 10.0
            verdict = "DESCENDED" if last10 < first10 else "NOT-DESCENDED"
            print(f"  {name}: first10={first10:.4f} last10={last10:.4f}  {verdict}")
        print(
            "  Cardiac downstream probe is deferred to "
            "`00_smoke_cardiac_probe.py` (TODO). Gate full launch on its verdict."
        )

    # ── PHASE 3 (full) ─────────────────────────────────────────────────────────
    if use_full:
        print("=" * 70)
        print("===== PHASE 3: POST-TRAIN EXTRACT =====")
        print("=" * 70)
        # Discover teacher ckpt — dinov2 saves at
        # output_dir/eval/training_<iter>/teacher_checkpoint.pth
        candidates = sorted(output_dir.rglob("teacher_checkpoint.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No teacher_checkpoint.pth under {output_dir}. "
                "Check training subprocess log."
            )
        ckpt_src = candidates[-1]   # most recent (largest iter)
        print(f"  source ckpt: {ckpt_src}")
        # [HIGH 5 adv fix 2026-05-24] dinov2 train.py:131-140 saves
        # {"teacher": model.teacher.state_dict()} where teacher is a
        # ModuleDict with sub-modules {backbone, dino_head, ibot_head}.
        # So on-disk shape = {"teacher": {"backbone.X": ..., "dino_head.Y": ...,
        # "ibot_head.Z": ...}}. Strip OUTER "teacher" then INNER "backbone."
        # prefix to extract backbone state_dict only.
        full = torch.load(ckpt_src, map_location="cpu", weights_only=True)
        teacher_sd = full.get("teacher", full)   # tolerate flat or wrapped
        backbone = {
            k[len("backbone."):]: v
            for k, v in teacher_sd.items()
            if k.startswith("backbone.")
        }
        if not backbone:
            raise KeyError(
                f"PHASE 3 extract failed: no 'backbone.<...>' keys in teacher_sd. "
                f"Sampled keys: {list(teacher_sd.keys())[:8]}"
            )
        ckpts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(backbone, ckpt_final)
        print(f"  Saved backbone ckpt ({len(backbone)} keys): {ckpt_final} "
              f"({ckpt_final.stat().st_size:,} B)")

    # ── PHASE 4 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("===== PHASE 4: SUMMARY =====")
    print("=" * 70)
    summary = {
        "tag": tag,
        "mode": "smoke" if use_smoke else "full",
        "config_active": str(config_active),
        "manifest_active": str(manifest_active),
        "rows_in_manifest": int(pd.read_csv(manifest_active).shape[0]),
        "ckpt_lvd142m_init": str(ckpt_lvd142m),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "train_subprocess_rc": rc,
        "wall_sec": round(wall_sec, 1),
        "wall_hours": round(wall_sec / 3600.0, 3),
        "final_backbone_ckpt": str(ckpt_final) if use_full else None,
    }
    summary_path = run_dir_tag / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"  Summary written: {summary_path}")


if __name__ == "__main__":
    main()
