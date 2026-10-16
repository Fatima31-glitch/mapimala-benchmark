"""Step 7 - Per-image inference latency of every saved model on a CPU (batch size 1, median and 90th percentile).

In the paper (Table 7) all models were timed in a single session on the two-core Google Colab CPU runtime (TF 2.21).
Run it on the CPU you want to characterise, after copying models/ and results/split.json there. Only the test images are needed. Output: results/latency_cpu.csv (+ hardware description).

Usage
  python 07_latency_cpu.py --data DIR            # all models in models/
  python 07_latency_cpu.py --data DIR --n 200 --threads 4
"""
import argparse
import csv
import glob
import os
import platform
import time
from pathlib import Path

import numpy as np

from common import MODELS, RESULTS, get_split, load_images, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200, help="test images timed (batch size 1)")
    ap.add_argument("--threads", type=int, default=0, help="limit TensorFlow to this many threads (0 = default)")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    import tensorflow as tf
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")  # CPU only, even if a GPU is present

    classes, paths, labels, idx = get_split(args.data)
    rng = np.random.default_rng(0)
    sel = rng.choice(idx["test"], min(args.n, len(idx["test"])), replace=False)
    x = load_images(paths[sel], 224, cache_key=None).astype(np.float32)

    hw = f"{platform.processor() or platform.machine()} | {os.cpu_count()} logical CPUs | {platform.system()} {platform.release()} | TF {tf.__version__}"
    print("hardware:", hw)
    rows = []
    for mp in (args.models or sorted(glob.glob(str(MODELS / "*.keras")))):
        name = Path(mp).stem
        model = load_model(mp)
        for i in range(10):  # warm-up
            model.predict(x[i:i + 1], verbose=0)
        times = []
        for i in range(len(x)):
            t0 = time.perf_counter(); model.predict(x[i:i + 1], verbose=0); times.append((time.perf_counter() - t0) * 1000)
        med, p90 = float(np.median(times)), float(np.percentile(times, 90))
        rows.append([name, f"{med:.1f}", f"{p90:.1f}", len(x)])
        print(f"[{name}] median={med:.1f} ms  p90={p90:.1f} ms")
    with open(RESULTS / "latency_cpu.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "latency_ms_median", "latency_ms_p90", "n_images"]); w.writerows(rows)
        w.writerow(["hardware", hw, "", ""])
    print("->", RESULTS / "latency_cpu.csv")


if __name__ == "__main__":
    main()
