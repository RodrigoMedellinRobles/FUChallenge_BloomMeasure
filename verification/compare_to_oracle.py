#!/usr/bin/env python3
"""compare_to_oracle.py — measure pixel displacement between the CONTAINER
output and the included 41_D7 oracle (avg_MRE 22.5822).

Only numpy is required, so this can run on the host or INSIDE the container.

Usage:
    python3 compare_to_oracle.py --pred out/regression_predictions.json \
                                 --oracle oracle_d7.json

Output: per-task table (mean/p99/max in px), verdict, and exit code.
    exit 0 -> reproduces the oracle (max < 0.5 px)
    exit 1 -> measurable but tolerable drift (max < 3 px)
    exit 2 -> large drift, DO NOT submit
    exit 3 -> structural error (missing images, mismatched task_id, etc.)
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="container regression_predictions.json")
    ap.add_argument("--oracle", required=True, help="included oracle_d7.json reference (22.5822)")
    a = ap.parse_args()

    pred_list = json.load(open(a.pred))
    ref = json.load(open(a.oracle))
    pred = {r["image_path"]: r for r in pred_list}

    print(f"container: {len(pred_list)} predictions ({len(pred)} unique image_path values)")
    print(f"oracle   : {len(ref)} predictions")

    # --- structural checks (fail BEFORE comparing numbers) ---
    fatal = []
    if len(pred_list) != len(pred):
        fatal.append(f"DUPLICATE image_path values in output: {len(pred_list)-len(pred)} duplicates")
    missing = [r["image_path"] for r in ref if r["image_path"] not in pred]
    extra = set(pred) - {r["image_path"] for r in ref}
    if missing:
        fatal.append(f"MISSING {len(missing)} images from the container output. "
                     f"First 5: {missing[:5]}")
    if extra:
        fatal.append(f"EXTRA {len(extra)} images not found in the oracle: {sorted(extra)[:5]}")
    if fatal:
        print("\n>>> STRUCTURAL ERROR")
        for f in fatal:
            print("   -", f)
        print("\nThis is NOT a precision issue: the container lost or created rows.")
        return 3

    # --- numeric diff ---
    per_task, counts, worst = {}, {}, (0.0, None)
    for r in ref:
        ip, tid = r["image_path"], r["task_id"]
        if pred[ip]["task_id"] != tid:
            print(f"\n>>> ERROR: task_id mismatch for {ip}: "
                  f"container={pred[ip]['task_id']} oracle={tid}")
            return 3
        x = np.asarray(pred[ip]["predicted_points_pixels"], float)
        y = np.asarray(r["predicted_points_pixels"], float)
        if x.shape != y.shape:
            print(f"\n>>> ERROR: different length for {ip}: {x.shape} vs {y.shape} "
                  f"(incorrect num_classes)")
            return 3
        d = np.linalg.norm(x.reshape(-1, 2) - y.reshape(-1, 2), axis=1)
        per_task.setdefault(tid, []).extend(d.tolist())
        counts[tid] = counts.get(tid, 0) + 1
        if d.max() > worst[0]:
            worst = (float(d.max()), ip)

    print(f"\n{'task':<14}{'n_img':>7}{'mean_px':>12}{'p99_px':>12}{'max_px':>12}")
    print("-" * 57)
    allv = []
    for t in sorted(per_task):
        v = np.array(per_task[t])
        allv += v.tolist()
        print(f"{t:<14}{counts[t]:>7}{v.mean():12.6f}{np.percentile(v,99):12.6f}{v.max():12.6f}")
    allv = np.array(allv)
    print("-" * 57)
    print(f"{'GLOBAL':<14}{len(ref):>7}{allv.mean():12.6f}"
          f"{np.percentile(allv,99):12.6f}{allv.max():12.6f}")
    print(f"\npoints compared: {len(allv)}    worst image: {worst[1]} ({worst[0]:.6f} px)")

    print("\nREFERENCE — this is a coordinate-level comparison.")
    print("The person running it should record the environment of the run being compared.")

    if allv.max() < 0.5:
        print("\n>>> VERDICT: the container REPRODUCES the oracle (max < 0.5 px). OK to submit.")
        return 0
    if allv.max() < 3.0:
        print("\n>>> VERDICT: small drift (< 3 px). Acceptable, but record the value:")
        print("    the container stack (torch 2.3.1 / py3.10) is NOT numerically")
        print("    identical to the validated stack. Check that no task concentrates")
        print("    the drift before submitting.")
        return 1
    print("\n>>> WARNING: LARGE drift (> 3 px). DO NOT submit. Check the")
    print("    torch / timm / albumentations / scipy versions inside the image (step 02).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
