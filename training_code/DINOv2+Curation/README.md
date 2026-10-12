# DINOv2+Curation

Internal tag `m2_v4`. Checkpoint inside the image: `/app/weights/m2_v4.pth`.

Same architecture and schedule as DINOv2+Cons, but the unlabeled corpus is routed instead of
pooled: every unlabeled image is scored first and sent to pseudo-labeling, consistency-only, or
discard. `METHOD.md` section 3 describes the five signals and the routing rule.

| File | What it is |
|---|---|
| `run_v4_umv2.sh` | What was run. Sets `BATCH=32` and `LR_REF=32`, then calls the engine three times: build, smoke, full |
| `nb06_train_r2_engine.py` | The engine, in the sense explained below |
| `hc_res700_flat_engine.py` | Imported by the engine at module level, for the HC-at-700 patch strings it asserts byte-identity against |
| `summary.json` | The run record, including the routing counts: 27,122 pseudo-labeled, 131,330 consistency-only, 32,717 discarded |

    ./run_v4_umv2.sh

It writes its progress into a `runs/` directory beside itself, so it leaves logs behind even when
it stops early.

## This member has no training loop of its own

`nb06_train_r2_engine.py` is a patcher. It reads DINOv2+Cons's engine
(`../DINOv2+Cons/nb04_train_unimatch_conf_only.py` in this repository, an internal repository path in
the original), applies twelve string substitutions and executes the result. Eight substitutions
are its own; four more are added because the launcher passes `--umv2`. So this member is
DINOv2+Cons plus twelve patches plus the curation manifest, and that is the most compact honest
statement of what it is.

The header of `run_v4_umv2.sh` discusses a batch size of 96 and the learning-rate scaling that
would follow. The variables below it set 32, which is what ran. The header is stale and we left the
file unedited rather than risk touching a file that produced a shipped checkpoint.
