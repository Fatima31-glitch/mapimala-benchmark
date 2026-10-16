"""Step 1 - Extract frozen ImageNet features (global-average-pooled) for every backbone.

Outputs
  features/<backbone>.npz          : train/val/test feature matrices + labels
  results/backbone_cost.json       : parameters, extraction latency (ms/image, batch 1) per backbone

Usage
  python 01_extract_features.py --data /path/to/DIMPSAR_40 [--backbones vgg16,mobilenetv3l,...] [--limit N]
"""
import argparse
import time

import numpy as np

from common import (BACKBONES, FEATURES, RESULTS, build_backbone, count_params, get_split, latency_ms, load_images,
                    load_json, preprocess_np, save_json, set_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder with one sub-folder per species")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (smoke tests only)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images of each subset (smoke tests)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    set_seeds(42)
    weights = None if args.weights == "none" else args.weights

    classes, paths, labels, idx = get_split(args.data)
    if args.limit:
        idx = {k: v[: args.limit] for k, v in idx.items()}
    x = {k: load_images(paths[v], 224, cache_key=None if args.limit else k) for k, v in idx.items()}
    y = {k: labels[v] for k, v in idx.items()}

    cost = load_json(RESULTS / "backbone_cost.json") if (RESULTS / "backbone_cost.json").exists() else {}
    for key in args.backbones.split(","):
        key = key.strip()
        print(f"\n=== {key} ===")
        bb, mode = build_backbone(key, weights)
        feats = {}
        t0 = time.time()
        for split in ("train", "val", "test"):
            xp = preprocess_np(x[split], mode)
            feats[split] = bb.predict(xp, batch_size=args.batch, verbose=0).astype(np.float32)
            print(f"  {split}: {feats[split].shape}")
        extract_time = time.time() - t0
        np.savez_compressed(FEATURES / f"{key}.npz", **{f"x_{k}": v for k, v in feats.items()},
                            **{f"y_{k}": y[k] for k in feats}, classes=np.array(classes))
        lat = latency_ms(lambda a: bb.predict(preprocess_np(a, mode), verbose=0), x["test"], n=100)
        cost[key] = {**count_params(bb), "feature_dim": int(feats["test"].shape[1]),
                     "extract_time_s_all_splits": round(extract_time, 1), "backbone_latency_ms": round(lat, 2)}
        print(f"  params={cost[key]['params_total']:,}  latency={lat:.1f} ms/image  extraction={extract_time:.0f}s")
        save_json(cost, RESULTS / "backbone_cost.json")


if __name__ == "__main__":
    main()
