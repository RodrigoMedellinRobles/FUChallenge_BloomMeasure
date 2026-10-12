# Third-party notices

| Component | Files in this repo | Origin | License |
|---|---|---|---|
| Organizer baseline code | `docker/vendor/baseline/{model.py,model_factory.py,utils.py}` (adapted) | github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry (official FU_Biometry baseline) | no license file declared in the upstream repository at the time of writing; used as the challenge-provided baseline, adapted |
| Organizer entry script | `docker/predict.py` (unmodified) | provided by the FU_Biometry organizers | same as above |
| UltraDINO / DINOv2 self-supervised training code | `training_code/DINOv2+SSL/ssl_code/` (config + LICENSE + UPSTREAM.md only; training code not redistributed) | github.com/jakobamb/UltraDINO, fork of github.com/facebookresearch/dinov2 | Apache License 2.0 |
| DINOv2 ViT-S/14 LVD-142M weights | not redistributed; used as initialization | Meta AI | Apache License 2.0 (as released by Meta) |
| USFM (BEiT-B/16 ultrasound) weights | not redistributed; used as initialization | github.com/openmedlab/USFM, Jiao et al., Medical Image Analysis 2024 | distributed by the original authors under their own terms |
| USFM encoder loader | `docker/vendor/usfm_src/usfm_encoder.py` | written by us around timm's BEiT implementation | MIT (ours) |
| DARK decoding | `docker/vendor/src/eval/dark_decoder.py` | our implementation of Zhang et al., CVPR 2020 | MIT (ours) |
| Python dependencies | `docker/requirements.txt` | PyPI | their own licenses |

Everything else is Bloom Standard Inc. code under MIT.
