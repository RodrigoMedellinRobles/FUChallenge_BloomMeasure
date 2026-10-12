# DINOv2+Cons-B

Internal tag `r3_a4c`. Checkpoint inside the image: `/app/weights/r3_a4c.pth`.

DINOv2+Cons with one change: A4C is excluded from the consistency loss. The A4C unlabeled pool is
two acquisition domains under one label (2,238 raw images at 42% Doppler, 61,152 pre-processed
patches at 0%), and consistency forces agreement between populations that should not agree. This
member enters the ensemble for one task only, fetal femur.

| File | What it is |
|---|---|
| `run_nb04v3_a4c_no_ssl.py` | What was run. Sets the A/B contract, the one knob (`exclude_cons_tasks`) and the falsification criterion, then imports the engine as a module |
| `nb04v3_train_a4c_no_ssl.py` | The engine |
| `config_snapshot.yaml` | Configuration of the run that produced the shipped checkpoint |

    python run_nb04v3_a4c_no_ssl.py
