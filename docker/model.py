"""model.py — the 41_D7 ensemble (avg_MRE 22.5822), packaged for the offline container.

Reproduces submissions/41_D7_quadv4_psax60_femur3dino/regression_predictions.json exactly
(the team's private-test oracle). Recipe verbatim from
experiments/FINAL_SUBMISSIONS/04_ensemble_variants/_models/build_ensemble_variant.py
invoked as:
    --members nb04 ultradino_ivc usfm m2_v4 --psax-blend 0.6 \
    --task-members fetal_femur:nb04,r3_a4c,ultradino_ivc   (FA geom on by default)

Per-task weighted average of 64x64 sigmoid heatmaps -> DARK (sigma=1.8) -> original pixels:
    A4C AOP FA FUGC HC IVC PLAX  ->  mean(nb04, ultradino_ivc, usfm, m2_v4)       (0.25 each)
    PSAX                          ->  0.6*usfm + 0.4*mean(nb04, ultradino_ivc, m2_v4)
    fetal_femur                   ->  mean(nb04, r3_a4c, ultradino_ivc)            (0.333 each)
    FA                            ->  + sub13 ellipse projection

All five models build and run in ONE process. Fully offline — weights are baked in.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

APP = Path(__file__).resolve().parent
for _p in ("vendor", "vendor/baseline", "vendor/folder04", "vendor/usfm_src"):
    p = str(APP / _p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_by_path(mod_name, rel_path):
    """Load a vendored .py by explicit path, to avoid the name clash between this
    file (model.py) and the organizer baseline's model.py (also 'model')."""
    spec = importlib.util.spec_from_file_location(mod_name, str(APP / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_baseline = _load_by_path("baseline_infer", "vendor/baseline/model.py")
InferenceDataset = _baseline.InferenceDataset
inference_collate_fn = _baseline.inference_collate_fn

TASK_NUMCLS = {"A4C": 16, "AOP": 4, "FA": 4, "FUGC": 2, "HC": 4,
               "IVC": 2, "PLAX": 22, "PSAX": 4, "fetal_femur": 2}
HEATMAP_SIZE = (64, 64)
SMOOTH_SIGMA = 1.8

# name -> (weight file in weights/, preprocess, input_size, kind)
MEMBERS = {
    "nb04":          ("nb04.pth",          "clahe_clip_sharpen_speckle",        518, "dino3"),
    "r3_a4c":        ("r3_a4c.pth",        "clahe_clip_sharpen_speckle",        518, "dino3"),
    "ultradino_ivc": ("ultradino_ivc.pth", "sharpen_speckle_1ch_ours_noclahe",  518, "dino1"),
    "m2_v4":         ("m2_v4.pth",         "clahe_clip_sharpen_speckle",        518, "dino3"),
    "usfm":          ("usfm.pth",          "clahe_clip_sharpen_speckle_beit05", 512, "usfm"),
}

# ---- 41_D7 per-task fusion ----
SEVEN = ["nb04", "ultradino_ivc", "usfm", "m2_v4"]         # A4C AOP FA FUGC HC IVC PLAX
FEMUR = ["nb04", "r3_a4c", "ultradino_ivc"]                # fetal_femur
PSAX_USFM_W = 0.6
PSAX_REST = ["nb04", "ultradino_ivc", "m2_v4"]             # the (1-w) split; excludes usfm

# which members each task consumes (drives what each model must forward)
TASK_MEMBERS = {t: SEVEN for t in ("A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX")}
TASK_MEMBERS["PSAX"] = ["usfm"] + PSAX_REST
TASK_MEMBERS["fetal_femur"] = FEMUR
FA_PROJECT_TASKS = {"FA"}

# FA ellipse constraint projector (sub13). C x = 0 for the 4 FA points (two perpendicular
# axes sharing a centre). P = I - C^T (C C^T)^-1 C is the orthogonal non-expansive projector.
_FA_C = np.array([[1, 0, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, -1],
                  [1, 0, 1, 0, -1, 0, -1, 0],
                  [0, 1, 0, 1, 0, -1, 0, -1]], dtype=float)
_FA_P = np.eye(8) - _FA_C.T @ np.linalg.inv(_FA_C @ _FA_C.T) @ _FA_C


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = APP / "weights"
        self.task_configs = [{"task_id": t, "task_name": "Regression", "num_classes": n}
                             for t, n in TASK_NUMCLS.items()]

    def _build(self, kind, ckpt):
        from model_factory import MultiTaskModelFactory
        if kind == "dino3":
            m = MultiTaskModelFactory(encoder_name="vit_small_patch14_dinov2.lvd142m",
                                      encoder_weights=None, task_configs=self.task_configs,
                                      heatmap_size=HEATMAP_SIZE).to(self.device)
            m.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        elif kind == "dino1":
            from model_factory_patched import MultiTaskModelFactory as M1
            # encoder_weights=None (NOT "pretrained"): the checkpoint is a full state_dict,
            # so no backbone download is needed — required for the offline container.
            m = M1(encoder_name="vit_small_patch14_dinov2.lvd142m", encoder_weights=None,
                   task_configs=self.task_configs, heatmap_size=HEATMAP_SIZE, in_chans=1).to(self.device)
            m.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        elif kind == "usfm":
            from usfm_encoder import attach_usfm_encoder
            m = MultiTaskModelFactory(encoder_name="beit_base_patch16_224", encoder_weights=None,
                                      task_configs=self.task_configs, heatmap_size=HEATMAP_SIZE).to(self.device)
            attach_usfm_encoder(m, img_size=512, weights_path=None, device=self.device,
                                load_weights=False, verbose=False)
            res = m.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True),
                                    strict=False)
            unexpected = [k for k in res.unexpected_keys if not k.startswith("relation.")]
            assert not res.missing_keys and not unexpected, \
                f"USFM load mismatch: missing={list(res.missing_keys)[:4]} unexpected={unexpected[:4]}"
        else:
            raise ValueError(kind)
        m.eval()
        return m

    @torch.no_grad()
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        from src.eval.dark_decoder import decode_heatmaps_dark
        from src.training.albu_transforms import build_transforms

        member_tasks = {name: set() for name in MEMBERS}
        for task, mems in TASK_MEMBERS.items():
            for mem in mems:
                member_tasks[mem].add(task)

        dumps = {name: {} for name in MEMBERS}   # member -> {image_path: (K,64,64) f32}
        meta = {}                                # image_path -> (task_id, (H, W))

        for name, (wfile, prep, insz, kind) in MEMBERS.items():
            tasks = member_tasks[name]
            if not tasks:
                continue
            ckpt = self.weights_dir / wfile
            assert ckpt.exists(), f"missing weight: {ckpt}"
            model = self._build(kind, str(ckpt))
            tf = build_transforms(preprocess=prep, input_size=insz, training=False)
            ds = InferenceDataset(data_root=data_root, transforms=tf, split_csv=None)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                                 num_workers=4, collate_fn=inference_collate_fn)
            for batch in loader:
                images = batch["image"].to(self.device)
                tids = batch["task_id"]
                paths = batch["image_path"]
                osizes = batch["original_size"]
                for tid in set(tids):
                    if tid not in tasks:
                        continue
                    idx = [i for i, t in enumerate(tids) if t == tid]
                    hm = torch.sigmoid(model(images[idx], task_id=tid)).cpu().numpy().astype(np.float32)
                    for k, b in enumerate(idx):
                        dumps[name][paths[b]] = hm[k]
                        meta[paths[b]] = (tid, osizes[b])
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[model] {name}: forwarded {len(dumps[name])} images", flush=True)

        results = []
        for path, (tid, osize) in meta.items():
            if tid == "PSAX":
                hm = PSAX_USFM_W * dumps["usfm"][path]
                hm = hm + (1.0 - PSAX_USFM_W) * np.mean([dumps[m][path] for m in PSAX_REST], axis=0)
            elif tid == "fetal_femur":
                hm = np.mean([dumps[m][path] for m in FEMUR], axis=0)
            else:
                hm = np.mean([dumps[m][path] for m in SEVEN], axis=0)

            norm = decode_heatmaps_dark(torch.from_numpy(hm[None].astype(np.float32)),
                                        smooth_sigma=SMOOTH_SIGMA).numpy()[0]
            H, W = int(osize[0]), int(osize[1])
            px = np.empty_like(norm)
            px[0::2] = norm[0::2] * W
            px[1::2] = norm[1::2] * H
            if tid in FA_PROJECT_TASKS:
                px = _FA_P @ px
                n = px.copy(); n[0::2] /= W; n[1::2] /= H
                n = np.clip(n, 0.0, 1.0)
                px = n.copy(); px[0::2] *= W; px[1::2] *= H
            results.append({"image_path": path, "task_id": tid,
                            "predicted_points_pixels": [float(v) for v in px]})

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "regression_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {out_path}", flush=True)
