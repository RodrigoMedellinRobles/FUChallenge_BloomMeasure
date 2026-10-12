# BloomMeasure: multi-task ultrasound landmark detection (FUB 2026, team Bloom Standard)

BloomMeasure is an ensemble of five landmark models. Their 64x64 heatmaps are averaged per task and decoded once with DARK, and FA is corrected by a closed-form ellipse projection. The internal name is D7 and the paper name is BloomMeasure.

| | |
|---|---|
| Team | Bloom Standard |
| Challenge | FU_Biometry / FUB 2026, MICCAI 2026 |
| Final test result | average MRE 22.6033, average MAE 27.3467 |
| Leaderboard rank | 13 of 31 on Leaderboard A, 9 of 31 on Leaderboard B |
| Paper | [Multi-Task Landmark Detection for Fetal and Cardiac Ultrasound Biometry under Label Scarcity](https://papers.miccai.org/miccai-2026-sat/FoundUS_092.html), MICCAI 2026 Workshops and Challenges (FoundUS satellite event, paper 92), LNCS 17261, Springer. [Open-access PDF](https://papers.miccai.org/miccai-2026-sat/paper/FoundUS_092.pdf), [OpenReview](https://openreview.net/forum?id=P9ojyliHnQ). DOI pending |
| Docker image | `flajucas/fu-biometry:v1.0.0`, digest `sha256:704004984a1a959e7757ac3e1d279cc13827ab4987c4c16c2acf2b4f076bafa1` |
| Contact | rodrigo@bloomstandard.com |

| Member | Role |
|---|---|
| Rodrigo Medellin-Robles | corresponding author, system design, experiments |
| Flavio A. Juarez-Castro | training infrastructure, curation pipeline, evaluation |
| Mohammad A. Bukhari | manuscript, literature review, analysis |

Code is released under the MIT License. Model weights are released as GitHub release assets (see Weights).

## Quick start: run the submitted container

```bash
docker pull flajucas/fu-biometry:v1.0.0
docker run --rm --gpus all --network none \
  --memory 7g --cpus 4 --shm-size 2g \
  -v /path/to/input:/input:ro -v /path/to/output:/output:rw \
  -e GU_INPUT_DIR=/input -e GU_OUTPUT_DIR=/output \
  flajucas/fu-biometry:v1.0.0
```

On 619 validation images this used 909 MiB of GPU memory, 1.97 GB of host memory and about four minutes. Input and output follow the challenge format through `GU_INPUT_DIR` and `GU_OUTPUT_DIR`.

### Runtime

The image is based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` (Python 3.10, torch 2.3.1, CUDA 12.1). Other Python pins are in `docker/requirements.txt`. A GPU is used when available and the code falls back to CPU otherwise (`docker/model.py`).

### Input and output format

`GU_INPUT_DIR` must contain `test_metadata.csv` with columns `image_path,task_id,task_name,num_classes` (the same header as `training_code/splits/train.csv`, with `task_name` set to `Regression`) and one directory per task holding the images, named `<task>_test/` or `<task>/` (see `docker/predict.py`). The container writes `GU_OUTPUT_DIR/regression_predictions.json`, a list of records `{image_path, task_id, predicted_points_pixels}` where `predicted_points_pixels` is a flat list `[x1, y1, x2, y2, ...]` in original image pixels (see `docker/model.py`).

### Run outside Docker

`docker/predict.py` hardcodes container paths (`/app`, `/work`) and is not meant to run on a host. Instead, with `docker/weights/` populated and `docker/requirements.txt` plus torch 2.3.1 installed, run from inside `docker/`:

```bash
cd docker
python3 -c "from model import Model; Model().predict(data_root='/path/to/data', output_dir='/path/to/out', batch_size=8)"
```

Here `data_root` holds `csv/*.csv` and `images/<task>/` as `docker/vendor/baseline/model.py` expects.

## Weights

The five checkpoints total 892 MB and are not in the git tree. Download the `nb04.pth`, `r3_a4c.pth`, `ultradino_ivc.pth`, `m2_v4.pth` and `usfm.pth` assets from the Releases page of this repository (the assets are attached to the latest release), then place them in `docker/weights/`. Alternatively, extract them from the public image:

```bash
cid=$(docker create flajucas/fu-biometry:v1.0.0)
docker cp "$cid":/app/weights ./docker/weights
docker rm "$cid"
```

Verify the files with:

```bash
(cd docker/weights && sha256sum -c ../../verification/weights.sha256)
```

With `docker/weights/` populated, `docker build -t bloommeasure docker/` runs to completion (the rebuilt image is functionally equivalent to the submitted one but not byte-identical, because `docker/vendor/` carries only the modules on the inference path; see `docker/vendor/INVENTORY.md`). `docker/best_model.pth` is a 1.3 kB stub required by the organizer entry script. Its payload says that the real weights are in `/app/weights/`.

## Verify that this code is the submitted system

Read the code and weight hashes from the image and compare them with `verification/CHECKSUMS.txt`:

```bash
docker run --rm --entrypoint sh flajucas/fu-biometry:v1.0.0 \
  -c 'md5sum /app/predict.py /app/model.py /app/requirements.txt && sha256sum /app/weights/*.pth'
```

Compare fresh predictions with the reference output:

```bash
python3 verification/compare_to_oracle.py \
  --pred /path/to/regression_predictions.json \
  --oracle verification/oracle_d7.json
```

The recorded run has 619 of 619 records present, mean drift 0.0030 px and maximum drift 0.088 px. The residual is consistent with stored member heatmaps being float16 while live code computes in float32. The checker exits 0 below 0.5 px maximum drift, 1 below 3 px, 2 above that, and 3 on structural failure.

## Results

| Task | MRE (px) | MAE |
|---|---:|---:|
| A4C | 24.31 | 18.33 |
| AOP | 17.32 | 6.12 |
| FA | 24.83 | 108.66 |
| fetal_femur | 15.69 | 15.38 |
| FUGC | 15.37 | 9.11 |
| HC | 24.55 | 50.66 |
| IVC | 22.15 | 7.25 |
| PLAX | 16.43 | 8.04 |
| PSAX | 42.78 | 22.57 |
| **Average** | **22.6033** | **27.3467** |

The test average of 22.60 and public validation average of 22.5822 are almost the same, but that is a coincidence rather than evidence that test and validation behave the same per task; the per-task breakdown shows two different regimes.

| Task | MRE on validation | MRE on test | Difference |
|---|---:|---:|---:|
| HC | 44.74 | 24.55 | -20.19 |
| PSAX | 34.53 | 42.78 | +8.25 |
| FUGC | 9.10 | 15.37 | +6.27 |
| AOP | 12.27 | 17.32 | +5.05 |
| FA | 20.03 | 24.83 | +4.80 |
| fetal_femur | 18.46 | 15.69 | -2.77 |
| PLAX | 17.54 | 16.43 | -1.11 |
| A4C | 24.37 | 24.31 | -0.06 |
| IVC | 22.20 | 22.15 | -0.05 |

Both leaderboards score the same predictions. Leaderboard A normalizes each metric per task across teams and averages the nine normalized values, producing MRE 0.21136, MAE 0.36310 and final score 0.28723. Leaderboard B averages raw metrics first and normalizes the two averages across teams, producing normalized values 0.08931 and 0.32957 and final score 0.20944.

## Names

| Paper | Code and checkpoint | Training code |
|---|---|---|
| DINOv2+Cons | `nb04`, `/app/weights/nb04.pth` | `training_code/DINOv2+Cons/` |
| DINOv2+Cons-B | `r3_a4c`, `/app/weights/r3_a4c.pth` | `training_code/DINOv2+Cons-B/` |
| DINOv2+Curation | `m2_v4`, `/app/weights/m2_v4.pth` | `training_code/DINOv2+Curation/` |
| DINOv2+SSL | `ultradino_ivc`, `/app/weights/ultradino_ivc.pth` | `training_code/DINOv2+SSL/` |
| USFM | `usfm`, `/app/weights/usfm.pth` | `training_code/USFM/` |

Paper names communicate model variants. Code names preserve the actual checkpoint filenames, dictionary keys and modules used by the submitted image.

| Weight | Source | Use |
|---|---|---|
| DINOv2 ViT-S/14, LVD-142M | publicly released by Meta AI | encoder initialization for four members |
| USFM, BEiT-B/16 ultrasound | publicly released with the USFM paper (Jiao et al., Medical Image Analysis 2024) | encoder initialization for the USFM member |

## Repository layout

```text
docker/                  Submitted inference code and vendored runtime modules
docker/submission/submission.json
                         Codabench submission descriptor, byte for byte as uploaded
training_code/           Recipes and artifacts for the five ensemble members
verification/            Checksums, reference predictions and comparison tool
METHOD.md                Detailed method description
docs/COMPLIANCE.md       Challenge-rules compliance and organizer clarification
THIRD_PARTY_NOTICES.md   Origins and licenses of third-party components
CONTRIBUTING.md          Branching, commits and pull-request requirements
CHANGELOG.md             Release history
CITATION.cff             Machine-readable citation metadata
```

## Training

`training_code/` documents the recipe for each member and is not a turnkey rerun. Scripts stop with `repo root not found` because they search for a `pyproject.toml`. A rerun also needs the challenge images, unlabeled corpus, public DINOv2 and USFM encoder weights, and curation manifest. See `training_code/README.md`.

## Method

`METHOD.md` describes the system in 13 sections. See `docs/COMPLIANCE.md` for the challenge-rules compliance statement and the organizer clarification on per-task fusion weights.

## Known stale comments

`docker/best_model.pth` is a 1.3 kB stub required because the organizer entry script copies that path unconditionally. The first comment in `docker/Dockerfile`, one line in `docker/requirements.txt`, and the header of `training_code/DINOv2+Curation/run_v4_umv2.sh` describe an earlier candidate; they remain untouched to preserve the submitted artifacts. Some headers use internal paths and the word "oracle" for the team's own reference predictions, not organizer ground truth.

The header of `docker/model.py` refers to internal repository paths and calls the reference prediction file "the team's private-test oracle"; that phrase means the team's own reference predictions for the test submission (`verification/oracle_d7.json` is the validation counterpart), not access to any labels.

## Data

`training_code/splits/*.csv` contains image paths, task ids and pseudonymized subject ids, with no landmark coordinates. `training_code/DINOv2+SSL/manifest_187k.csv.xz` is a file list of the unlabeled corpus, with no pixels or labels. `verification/oracle_d7.json` contains the team's own 619 validation predictions and no ground truth. No challenge images or annotations are redistributed.

## Citation

The entry below is the one published on the MICCAI open-access page; the volume, month and page fields will be completed by the publisher.

```bibtex
@InProceedings{MedRod_MultiTask_MICCAISAT2026,
  author    = {Medellin-Robles, Rodrigo AND Juárez-Castro, Flavio A. AND Bukhari, Mohammad Abdullah},
  title     = {{Multi-Task Landmark Detection for Fetal and Cardiac Ultrasound Biometry under Label Scarcity}},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 17261},
  month     = {pending},
  page      = {pending},
  url       = {https://papers.miccai.org/miccai-2026-sat/FoundUS_092.html}
}
```

Open-access PDF: https://papers.miccai.org/miccai-2026-sat/paper/FoundUS_092.pdf. OpenReview: https://openreview.net/forum?id=P9ojyliHnQ. See `CITATION.cff` for machine-readable metadata.

## License

Our code is licensed under MIT. Third-party components remain under their own licenses, as described in `THIRD_PARTY_NOTICES.md`.

## Contributing

See `CONTRIBUTING.md` for the branching model, commit convention and pull-request template.
