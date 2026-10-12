# DINOv2+Cons

Internal tag `nb04`. Checkpoint inside the image: `/app/weights/nb04.pth`.

The reference member, and the one the other three DINOv2 members are each a single departure
from. DINOv2 ViT-S/14 at 518, nine task heads, two-stage schedule (5 epochs with the encoder
frozen, then 35 with everything trainable at a 5x lower encoder learning rate), EMA teacher at
0.999, and weak-to-strong consistency over the unlabeled corpus gated per task at the 25th
percentile of teacher confidence.

| File | What it is |
|---|---|
| `nb04_train_unimatch_conf_only.py` | The engine. Contains the training loop, the consistency loss and the per-task gate |
| `config_snapshot.yaml` | The configuration of the run that produced the shipped checkpoint, written by the run itself |

It was launched from a notebook, which is not shipped: its stored outputs came from an earlier
execution than the shipped checkpoint, so it would have described the wrong run.
`config_snapshot.yaml` is the record instead.

This engine is also read and patched at runtime by DINOv2+Curation, so it is a dependency of that
member as well as the source of this one.
