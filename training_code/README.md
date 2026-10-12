# How the five checkpoints were produced

One directory per ensemble member, named as the paper names it.

| Paper name | Folder | Internal tag | Checkpoint | What it is |
|---|---|---|---|---|
| DINOv2+Cons | `DINOv2+Cons/` | nb04 | `nb04.pth` | the reference member |
| DINOv2+Cons-B | `DINOv2+Cons-B/` | r3_a4c | `r3_a4c.pth` | the reference with A4C out of the consistency loss |
| DINOv2+Curation | `DINOv2+Curation/` | m2_v4 | `m2_v4.pth` | the reference with the unlabeled corpus routed instead of pooled |
| DINOv2+SSL | `DINOv2+SSL/` | ultradino_ivc | `ultradino_ivc.pth` | encoder self-supervised on the challenge corpus, single channel |
| USFM | `USFM/` | usfm | `usfm.pth` | BEiT-B/16 from public ultrasound pre-training, supervised only |

Each folder is named as the paper names it, and carries a README explaining what that member is,
what was run and what each file in it does. The internal tag is the name used in `docker/model.py`
and in the checkpoint filename, so the chain from a paper name to a SHA-256 in
`verification/CHECKSUMS.txt` is unbroken in both directions.

Each folder holds the engine that did the training. Three of the five also hold the script that
launched it, which carries the configuration and the intent; the other two were launched from
notebooks that are not shipped, and their configuration is recorded in `config_snapshot.yaml`
instead.

## One member has no training loop of its own

`DINOv2+Curation/nb06_train_r2_engine.py` is a patcher, not a trainer. At its `NB04_SRC` block it
reads DINOv2+Cons's engine `nb04_train_unimatch_conf_only.py`, applies twelve string substitutions (eight in its base set, four more because the launcher passes `--umv2`), and `exec`s the result.
So DINOv2+Curation is DINOv2+Cons's engine plus those patches plus the curation manifest, and that
engine is a runtime dependency of this member, not merely a sibling. It also imports `hc_res700_flat_engine.py` at
module level, for the verbatim HC-at-700 patch strings it asserts byte-identity against; that file
ships here for that reason. Its padded sibling `hc_res700_engine.py` is not shipped: it is only
reached through a preflight gate this path never enters. The `res700-flat` name in `DINOv2+Curation/summary.json` is recipe lineage, not a second
training run.

## What the artifacts are for

`splits/train.csv` and `splits/internal_val.csv` define the labeled split, 5,386 images for
training and 1,357 held out for model selection. They carry image paths, task ids and subject ids
only; the landmark annotations belong to the organizers and are not redistributed here.

`DINOv2+Curation/summary.json` is the record of how the unlabeled corpus was routed: 27,122 images added as
pseudo-labels, 131,330 used for consistency only, 32,717 discarded, with a per-task breakdown.
`docs/COMPLIANCE.md` and `METHOD.md` section 3 cite those numbers, and this is the file
behind them.

`DINOv2+SSL/manifest_187k.csv.xz` is the corpus list for the self-supervised stage: 187,744
rows of `image_path,task_id,subdomain,bytes`. Decompress with `xz -d` and every path resolves to a
challenge task directory. `DINOv2+SSL/ours_norm_stats.json` reports the same corpus as 187,743 images plus 1
failure, which is the independent cross-check on that count.

`DINOv2+SSL/albu_transforms_TRAINING.py` is the training-side copy of the preprocessing module.
It differs from the container copy in `docker/vendor/src/training/`, and section 9 of `METHOD.md`
explains the difference and which set of grayscale statistics the submitted predictions were
produced under. It ships so the disclosure is checkable rather than asserted.

## Notes if you rerun any of this

The split changed size once. On 2026-07-23 the organizers withdrew 25 fetal_femur images as invalid, taking
the labeled pool from 6,768 to 6,743 and the split from 5,407 / 1,361 to the 5,386 / 1,357 shipped
here. Members trained before that date saw the larger version; no image moved between the two
portions.

Random seed 42 throughout the supervised stages. The self-supervised stage of DINOv2+SSL ran the upstream code, whose data sampler does not consume the run seed, so that stage is not bit-exact across reruns (noted in the shipped configuration file).

`VAL_ORG` in some engine assertions and `_corpus_nb04v2_no_valorg` in a config snapshot refer to a manifest of public validation images that was once considered as extra unlabeled data. No shipped member used it, and the engines assert its absence from the unlabeled pool.

Directory names such as `experiments_collaborator` in the config snapshots are internal run locations of a team member, kept so the records stay traceable.

Two notebooks are not shipped. The one that drove DINOv2+Cons held stored outputs from an execution that
predates both the split correction and the final stage-2 run, so it described a different run than
the shipped checkpoint; `DINOv2+Cons/config_snapshot.yaml` is the record of the run that produced it. The
one that drove the supervised stage of ultradino_ivc,
`nb01b_dapt_lpft_ema_seed42_noclahe_ep40.ipynb`, is 22 MB. Both are available on request.

This directory documents the recipe. It is not a turnkey rerun. Each script locates its repository root by walking up for a `pyproject.toml`, and there is none
above these files here, so running one directly stops immediately with `repo root not found`
(a `RuntimeError` in four of them, a `FileNotFoundError` naming the directory it searched from in
`DINOv2+SSL/nb01a_dapt_pretrain_seed42.py`). That is expected, not a broken package. A real
rerun would also need the challenge images, the unlabeled corpus, the public DINOv2 and USFM
encoder weights, and the curation manifest, none of which are ours to redistribute.

Two smaller things a reviewer diffing the tree will notice. The launcher header in
`DINOv2+Curation/run_v4_umv2.sh` discusses a batch size of 96, while the variables below it set 32, which is
what ran; the header is stale and we left the file unedited. And the single-channel model factory
used by DINOv2+SSL is not duplicated here: it ships once, at
`docker/vendor/folder04/model_factory_patched.py`.
