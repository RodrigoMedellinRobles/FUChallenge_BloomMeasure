# Compliance with the FUB 2026 challenge rules

## Semi-supervised use of the unlabeled data

Required, and used, by four of the five members, in two different ways. Three of them (DINOv2+Cons, DINOv2+Cons-B, DINOv2+Curation) train on the challenge unlabeled corpus through weak-to-strong consistency with an EMA teacher and a per-task confidence gate, and one of those three (DINOv2+Curation) additionally routes each unlabeled image into pseudo-label, consistency-only or discard. The fourth (DINOv2+SSL) uses the unlabeled corpus for self-supervised encoder pre-training on 187,744 images; its supervised stage carries no consistency branch.

Two facts are stated explicitly. First, the fifth member (USFM) uses no unlabeled data at any stage. Second, the routing in DINOv2+Curation discards 32,717 unlabeled images from that member's training. Neither reduces the corpus the submitted system consumed: the unlabeled data enters through the three consistency members, two of which use it whole while the third excludes A4C from consistency by design, and again through the self-supervised pre-training of DINOv2+SSL. Routing selects a treatment per image for one member rather than deciding whether the system sees the image at all.

## External data and pre-trained models

No external image dataset was used for training. No private annotation of any public data was produced or used. Two publicly released pre-trained encoders were used as initialization, both allowed by the rules:

| Weight | Source | Use |
|---|---|---|
| DINOv2 ViT-S/14, LVD-142M | publicly released by Meta AI | encoder initialization for four members |
| USFM, BEiT-B/16 ultrasound | publicly released with the USFM paper (Jiao et al., Medical Image Analysis 2024) | encoder initialization for the USFM member |

The self-supervised adaptation in DINOv2+SSL follows the UltraDINO methodology, but the pre-training itself was run by the team on challenge images. It is not the published UltraDINO checkpoint.

## Single unified solution

The submission is one Docker image, one entry point, one set of five checkpoints, and one inference pass that handles all nine tasks and all imaging domains. No checkpoint was trained for a single subgroup or task, and no separate submission was made per task.

Two tasks use different fusion weights over the same five checkpoints: PSAX weights the ultrasound pre-trained member at 0.60, and fetal femur averages three of the five members. The organizers confirmed on 2026-08-08 that different weights per task at inference do not count as subgroup-specific optimization as long as no checkpoint is trained for particular subgroups or tasks. No checkpoint here is.

## One test submission

One image was submitted to the test phase, Codabench submission 887183 on 2026-08-12. No other test submission was made.

## Offline execution

The container runs with `--network none`. All weights are copied into the image at build time, `pretrained=True` is never used at runtime, and the offline environment variables for Hugging Face and transformers are set in the Dockerfile.

The inference code was validated on 619 validation images against the reference prediction file, to 0.088 px maximum coordinate drift. The scripts in `verification/` reproduce that check. The image was built from `docker/`, run locally offline before submission, and pushed. The platform then ran it under `--network none` and returned a complete score for all nine tasks. `verification/compare_to_oracle.py` lets anyone repeat the comparison against the reference output.

## Code release

This repository is the public release of the code in `docker/` and `training_code/` that produced the submission.

## Paper

Published as paper 92 of the Foundation Model Challenge for Ultrasound Biometry (FoundUS) satellite event, in the MICCAI 2026 Workshops and Challenges proceedings (LNCS 17261, Springer; DOI pending). Open-access page: https://papers.miccai.org/miccai-2026-sat/FoundUS_092.html. OpenReview: https://openreview.net/forum?id=P9ojyliHnQ. The camera-ready was sent on 2026-08-11 through OpenReview. It states twice, in section 1 and section 6, that the system placed second in the validation phase. That was the public standing when it was written; the final standings are the ones in the repository README.

The abstract shown on the open-access web page is the one of the originally accepted submission (a single model at MRE 24.28); the PDF linked from that page is the camera-ready, which describes the five-member ensemble at 22.58 and matches this repository.

## Appendix: the clarification we asked for

Before submitting, we asked the organizers whether averaging several checkpoints inside one container is compliant, and whether the averaging weights may differ by task. Both questions were asked in writing on 2026-08-07 and answered on 2026-08-08.

### What we asked

> **Our setup.** We use several checkpoints. Each one is a multi-task model trained on all
> nine tasks and all imaging domains, none of them is trained for, or limited to, a subgroup.
> At inference, one pipeline runs them and averages their predicted heatmaps into a single
> output.
>
> **Question 1.** Is it allowed to put several checkpoints in the Docker container and average
> their outputs, if the same averaging is used for every task?
>
> **Question 2.** If yes, is it also allowed for the averaging weights to differ by task, for
> example weighting one checkpoint more for cardiac views than for fetal views? Or would that
> count as "subgroup-specific"?
>
> If Question 2 is outside the intended scope, we will use equal weights everywhere. We would
> rather be conservative than risk non-compliance.

### The reply we received

> Thanks for consulting our competition rules. For your questions:
>
> You can put multiple fully universal multi-task checkpoints in your Docker container and
> average their predictions equally for all tasks. All your checkpoints are trained on all 9
> tasks and all imaging domains, so equal ensemble is compliant.
>
> You can adopt different weights for different tasks during inference. This will not be
> counted as subgroup-specific optimization, as long as none of your checkpoints is only
> trained for some certain subgroups or tasks.
>
> We only restrict models customized and trained for independent subgroups. Inference-stage
> weight adjustment is not limited, but overly extreme weighting is not advised.

### How the submitted system relates to it

The condition is about training, and all five checkpoints are trained on all nine tasks and all imaging domains. None is task-specific or domain-specific.

On the advice against extreme weighting: a hard-routing variant that gave PSAX entirely to one member was measured and not submitted. The submitted system caps that member at 0.60 and leaves 0.40 with the other three, and fetal femur is an equal average of three members rather than a single one. Seven of the nine tasks are a plain equal average of four members.
