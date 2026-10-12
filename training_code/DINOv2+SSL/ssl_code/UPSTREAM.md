# Self-supervised pre-training code

The self-supervised stage of DINOv2+SSL was run with the public UltraDINO release,
"General methods make great domain-specific foundation models" (MICCAI 2025), which is itself a
fork of Meta's DINOv2. It is Apache 2.0; the licence travels with this directory.

We did not modify the training code. Our changes were configuration files under
`dinov2/configs/train/`, and the one that produced the shipped checkpoint is included here:
`ours_vits14_ibot_dapt.yaml`. It sets our encoder (ViT-S/14 warm-started from LVD-142M,
`in_chans=1`, no register tokens) and our corpus (the 187,744 challenge ultrasound images listed
in `../manifest_187k.csv.xz`). We also wrote short smoke variants and two arms we did not ship, a
three-channel one and a ViT-B with register tokens; they are not included because they produced
nothing in the submitted system.

The rest of the upstream tree is not included: it is 340 files of unmodified upstream
code, most of it downstream evaluation harnesses (classification, segmentation, depth) that the
pre-training stage never executes. To verify that our copy was unmodified, take the upstream
release and compare. For reference, the tree as we held it hashes to:

    sha256 of `tar -cf - ssl_code`   b993051cea96d799c391f85d2b7d8c6c893c26e93121421412a594a5795fdfe4
    files                            340

Entry point: `../nb01a_dapt_pretrain_seed42.py`. It reads
`dinov2/configs/train/ours_vits14_ibot_dapt.yaml` from the upstream checkout in our original
repository; the copy here is that same file, kept under the same relative path.
