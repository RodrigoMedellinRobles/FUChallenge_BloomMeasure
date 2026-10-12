# What is in this directory

Four files, for one purpose: this lets anyone confirm that the code in this repository is the code
that ran inside the submitted image.

**`CHECKSUMS.txt`** holds the SHA-256 of the five model checkpoints and the md5 of the four code
files in `docker/`. The checkpoints are not shipped, so these hashes are the only identity the
system has here. Read them out of the image with

    docker run --rm --entrypoint sh flajucas/fu-biometry:v1.0.0 \
      -c 'md5sum /app/predict.py /app/model.py /app/requirements.txt && sha256sum /app/weights/*.pth'

The `predict.py` md5 should equal the organizer entry script; ours was never modified.

**`weights.sha256`** is the five checkpoint lines of `CHECKSUMS.txt` in `sha256sum -c` format, for use after downloading the weights.

**`oracle_d7.json`** is our own output: the 619 predictions our system produced on the public
validation set, which scored average MRE 22.5822. It contains no ground truth. Several code
comments in this repository call it "the oracle"; that is internal shorthand for our reference
output, and we had no access to validation or test labels at any point.

sha256 `2bf2d177c7befc60933c2eed1ec8699fdd7e6584f0257a3b7d4c8271decd3a33`

Each record carries `image_path`, `task_id`, `predicted_points_pixels` and a fourth key,
`predicted_points_normalized`, which is ours for internal comparison. The container writes only
the three fields the challenge output format defines.

**`compare_to_oracle.py`** compares a prediction file against that reference, coordinate by
coordinate, after checking structure first: duplicate paths, missing or extra records, task
mismatches and wrong point counts each fail before any number is compared. It needs only numpy.

    python3 compare_to_oracle.py --pred /path/to/regression_predictions.json \
                                 --oracle oracle_d7.json

It exits 0 under 0.5 px of maximum drift, 1 under 3 px, 2 above, and 3 on a structural failure.
Our own run of the shipped code outside the container over the 619 images: 619 of 619 records
present, mean drift 0.0030 px, maximum 0.088 px, exit 0. The residual is consistent with the
stored member heatmaps being float16 while the live code computes in float32.

## How the reference file was produced

Five checkpoints, each forwarded once over the 619 images, producing one 64x64 sigmoid heatmap per
landmark. The maps are averaged per task with the weights in `METHOD.md` section 7, decoded once
with DARK at sigma 1.8 into original pixel coordinates, and the FA predictions are then passed
through the closed-form ellipse projection. That is the same path `docker/model.py` runs inside the
container.
