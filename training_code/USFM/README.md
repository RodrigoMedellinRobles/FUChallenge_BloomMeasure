# USFM

Internal tag `usfm`. Checkpoint inside the image: `/app/weights/usfm.pth`.

The only non-DINOv2 member: BEiT-B/16 initialized from the public USFM ultrasound pre-trained
weights, at 512 with a 32x32 patch grid and mean = std = 0.5. Supervised only, with no consistency
branch and no unlabeled data. It is the weakest member alone, and it is in the ensemble because
PSAX fails from a lack of in-domain information rather than from the decoder or the training
signal: an ultrasound pre-trained encoder moves that task from 42.16 to 34.58 px (hard-routed to USFM alone; the submitted 0.60 blend scores 34.53 on validation).

| File | What it is |
|---|---|
| `_run_usfm_both.py` | What was run. Handles the smoke pass and the full run for either normalization variant |
| `nb_usfm_train.py` | The engine |
| `_dark_eval_usfm.py` | Scoring pass the launcher imports and calls after training |
| `summary.json` | The run record |

    python _run_usfm_both.py --mode full --variants beit05

`beit05` is the variant that produced the shipped checkpoint. The other variant, ImageNet
normalization, was the A/B arm and is not part of the submitted system.
