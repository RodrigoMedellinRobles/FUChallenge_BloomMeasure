# DINOv2+SSL

Internal tag `ultradino_ivc`. Checkpoint inside the image: `/app/weights/ultradino_ivc.pth`.

The only member whose encoder was pre-trained by us. It continues DINOv2 self-supervised
pre-training (DINO, iBOT, KoLeo) for 40 epochs on the 187,744 unlabeled challenge images, with the
patch embedding folded to a single channel and no contrast equalization, and is then fine-tuned
with the same supervised recipe as the other DINOv2 members.

It is not the published UltraDINO checkpoint. We followed that methodology and ran the
pre-training ourselves on the challenge corpus.

Two stages, in order:

| File | What it is |
|---|---|
| `nb01a_dapt_pretrain_seed42.py` | Stage 1, the self-supervised pre-training. It reads a configuration named `dinov2/configs/train/ours_vits14_ibot_dapt.yaml`, from the upstream tree in our original repository |
| `nb01b_dapt_lpft_ema_seed42.py` | Stage 2, the supervised fine-tuning engine. It was launched from a notebook that is not shipped, 22 MB, available on request |
| `manifest_187k.csv.xz` | The corpus for stage 1: 187,744 rows of `image_path,task_id,subdomain,bytes`. Decompress with `xz -d`; every path resolves to a challenge task directory |
| `ours_norm_stats.json` | The grayscale statistics this member was trained under, and an independent count of the same corpus (187,743 images plus one failure) |
| `albu_transforms_TRAINING.py` | The training-side preprocessing module. It differs from the container copy, and `METHOD.md` section 9 explains how and which statistics the submitted predictions were produced under. It ships so that disclosure is checkable |
| `summary.json` | The run record |
| `ssl_code/` | See below |

## Why `ssl_code/` looks nearly empty

Stage 1 ran on the public UltraDINO release, which is a fork of Meta's DINOv2 under Apache 2.0. We
did not modify that code, so shipping it back would be several thousand files of somebody else's
work. What is here is the licence, a note recording which release it was and that it was
unmodified, and the one configuration file we wrote, which is what actually defines our run.

The config sits under `ssl_code/dinov2/configs/train/` rather than at the top of this directory
because that is where it lives inside the upstream tree, which is the trailing path
`nb01a_dapt_pretrain_seed42.py` resolves. Keeping the layout makes the correspondence obvious;
the script itself points at the upstream checkout in our original repository, not at this copy.
