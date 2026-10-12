# Third-party notices

Bloom Standard Inc. code in this repository is released under the MIT License (see LICENSE). The components below are not ours and are not covered by that grant.

| Component | Files in this repo | Origin | License |
|---|---|---|---|
| Organizer baseline code | `docker/vendor/baseline/model.py`, `model_factory.py`, `utils.py` (adapted) | github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry (official FU_Biometry baseline) | No license file in the upstream repository. Its README states: "This baseline code is for competition use only." See the note below. |
| Organizer entry script | `docker/predict.py` (unmodified) | github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry (official FU_Biometry baseline) | No license file in the upstream repository. Its README states: "This baseline code is for competition use only." See the note below. |
| UltraDINO / DINOv2 self-supervised training code | `training_code/DINOv2+SSL/ssl_code/` holds only the configuration file we changed, the upstream LICENSE and a provenance note (UPSTREAM.md); the training code itself is not redistributed | github.com/jakobamb/UltraDINO, fork of github.com/facebookresearch/dinov2 | Apache License 2.0 |
| DINOv2 ViT-S/14 LVD-142M weights | not redistributed; used as initialization | Meta AI | Apache License 2.0 (as released by Meta) |
| USFM (BEiT-B/16 ultrasound) weights | not redistributed; used as initialization | github.com/openmedlab/USFM, Jiao et al., Medical Image Analysis 2024 | distributed by the original authors under their own terms |
| USFM encoder loader | `docker/vendor/usfm_src/usfm_encoder.py` | written by us around timm's BEiT implementation | MIT (ours) |
| DARK decoding | `docker/vendor/src/eval/dark_decoder.py` | our implementation of Zhang et al., CVPR 2020 | MIT (ours) |
| Python dependencies | `docker/requirements.txt` | PyPI | their own licenses |

## Note on the organizer baseline files

The four organizer files listed above are redistributed here because the FU_Biometry challenge rules require participants to publish the code of their submission, and the submitted container runs these files unmodified or lightly adapted. They remain the organizers' code under the organizers' terms; the MIT license of this repository does not apply to them. Anyone reusing them outside the challenge should contact the organizers.
