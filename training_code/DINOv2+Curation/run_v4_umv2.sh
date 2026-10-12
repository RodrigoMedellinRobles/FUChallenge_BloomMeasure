#!/usr/bin/env bash
# ============================================================================
# folder-08 M2 — v4. The baseline is v3 (LB 23.7828, the project's best MODEL).
#
# ONE question: the ambiguous bucket changes from confidence-only UniMatch to
# FULL UniMatch v2 — strong(s1) + strong(s2) + feature perturbation on the weak
# view, all against the SAME EMA-teacher target and confidence mask.
# cons_weight /= 3 keeps the intended total consistency weight equal to v3:
# this tests whether the streams EXIST, not a three-times-larger term.
#
# Reuses v3 nb00..nb05b UNCHANGED: the same manifest, the same 2,262 A4C/IVC
# pseudo-labels, and the same 16,554 exclusions. The data root is rebuilt identically.
#
# WARNING: TWO VARIABLES, not one. The batch was explicitly increased to use
# available VRAM (32 -> 96), and the engine scales learning rates by sqrt(k) =
# x4.90. Thus v4 differs from v3 in (a) UMv2 and (b) batch+LR. Measurements on
# this GPU show that batch size adds no speed (12.1 -> 13.3 img/s from b=4 to
# b=64 under a 300 W power cap) and does not change exposure (= N/9; batch
# cancels out). This is recorded here for an honest interpretation of results.
#
# enc_lr becomes 4.90e-5, about 5x the champion's. This is the run's real risk:
# if stage 1 is much worse than v3's 44.38, the LR—not UMv2—is responsible.
#
# Launch:  setsid nohup ./run_v4_umv2.sh > /dev/null 2>&1 &
# Progress: cat runs/_STATUS_v4     Live: tail -f runs/_chain_v4.log
# Stop:     kill -- -$(cat runs/_chain_v4.pgid)
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
R="$HERE/runs"
LOG="$R/_chain_v4.log"
STATUS="$R/_STATUS_v4"
mkdir -p "$R"

TAG=M2_umv2_v4_0728
SRC_MANIFEST=M2_nocap_v3_0728      # unchanged v3 manifest
SRC_NB00=M2_nocap_v2_0727
BATCH=32   # Do not increase: at fixed epochs, batch 96 gives 3x FEWER optimizer updates (338 vs 1015 steps/ep) in the SAME wall time. Measured stage 1: 44.38 -> 77.03
LR_REF=32   # = BATCH, preserving the exact v3 learning rates

export R2_MANIFEST_DIR="$R/05b_rebalance_$SRC_MANIFEST"
export R2_NB00_DIR="$R/00_infer_$SRC_NB00"
export R2_OUT_ROOT="$R/06_train_$TAG"

echo $$ > "$R/_chain_v4.pgid"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG" "$STATUS" >/dev/null; echo "[$(ts)] $*"; }

stage() {
  local name="$1"; shift
  local marker="$1"; shift
  if [[ -e "$marker" ]]; then say "SKIP  $name (already done)"; return 0; fi
  say "START $name"
  local t0=$SECONDS
  if ! "$@" >>"$LOG" 2>&1; then
    say "FAIL  $name after $((SECONDS-t0))s -- see $LOG"; say "CHAIN ABORTED"; exit 1
  fi
  if [[ ! -e "$marker" ]]; then
    say "FAIL  $name exited 0 but did not produce $marker"; say "CHAIN ABORTED"; exit 1
  fi
  say "OK    $name  ($((SECONDS-t0))s)"
}

say "===== v4 start (pid $$) — ambiguous: conf-only -> FULL UniMatch v2 (s1+s2+FP p=0.5, cons_weight/3), batch $BATCH = full v3 parity, bf16, reused v3 manifest ====="

stage nb06_build "$R/06_train_$TAG/_dataroot_r2/csv/round2_train_labeled.csv" \
  "$PY" "$HERE/nb06_train_r2_engine.py" --build-only

stage nb06_smoke "$R/06_train_$TAG/00_SMOKE_$TAG/summary.json" \
  "$PY" "$HERE/nb06_train_r2_engine.py" --smoke --umv2 --tag "00_SMOKE_$TAG" --batch-size "$BATCH" --lr-ref-batch "$LR_REF"

stage nb06_full "$R/06_train_$TAG/01_full_$TAG/summary.json" \
  "$PY" "$HERE/nb06_train_r2_engine.py" --umv2 --tag "01_full_$TAG" --batch-size "$BATCH" --lr-ref-batch "$LR_REF"

say "===== CHAIN COMPLETE ====="
say "ckpt: $R/06_train_$TAG/01_full_$TAG/stage2_unfreeze.pth"
say "NEXT: evaluate original-pixel results against v3 (internal per-image 24.3752 with geometry / LB 23.7828)"
