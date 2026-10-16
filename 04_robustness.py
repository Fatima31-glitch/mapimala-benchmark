"""Step 4 - DIMPSAR-C: robustness of every saved model to smartphone-acquisition corruptions (test set only).

Nine corruptions x three severities (ImageNet-C style). Models are NOT retrained.
  gaussian_blur, motion_blur, gaussian_noise, jpeg, brightness, contrast, rotation, zoom, occlusion

Outputs
  results/robustness.csv            (model, corruption, severity, accuracy)
  results/robustness_summary.csv    (model, clean, mCA, relative robustness, worst corruption)
  cache/dimpsar_c/<corruption>_<severity>.npy  (the corrupted test images, released with the paper)

Usage
  python 04_robustness.py --data DIR [--models models/*.keras] [--limit N]
"""
import argparse
import csv
import glob
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from common import CACHE, RESULTS, get_split, load_images, load_model

SEVERITIES = {
    "gaussian_blur": [1, 2, 3],            # sigma (px at 224x224)
    "motion_blur": [5, 9, 15],             # kernel length (px)
    "gaussian_noise": [0.02, 0.05, 0.10],  # sigma on [0,1] scale
    "jpeg": [50, 30, 10],                  # quality
    "brightness": [0.2, 0.35, 0.5],        # +/- fraction (alternating sign per image)
    "contrast": [0.8, 0.6, 0.4],           # factor
    "rotation": [15, 30, 45],              # degrees (alternating sign per image)
    "zoom": [0.10, 0.20, 0.30],            # fraction of the field of view cropped
    "occlusion": [0.10, 0.20, 0.30],       # fraction of the area covered by a rectangle
}


def corrupt(img: np.ndarray, kind: str, sev, rng: np.random.Generator, i: int) -> np.ndarray:
    pil = Image.fromarray(img)
    sign = 1 if i % 2 == 0 else -1
    if kind == "gaussian_blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(sev)))
    if kind == "motion_blur":
        k = np.zeros((sev, sev), dtype=np.float32); k[sev // 2, :] = 1.0 / sev
        k = ndimage.rotate(k, rng.uniform(0, 180), reshape=False, order=1); k /= k.sum()
        out = np.stack([ndimage.convolve(img[..., c].astype(np.float32), k, mode="reflect") for c in range(3)], -1)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "gaussian_noise":
        out = img.astype(np.float32) / 255.0 + rng.normal(0, sev, img.shape)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=sev); buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "brightness":
        return np.asarray(ImageEnhance.Brightness(pil).enhance(1 + sign * sev))
    if kind == "contrast":
        return np.asarray(ImageEnhance.Contrast(pil).enhance(sev))
    if kind == "rotation":
        return np.asarray(pil.rotate(sign * sev, resample=Image.BILINEAR, fillcolor=tuple(int(v) for v in img.reshape(-1, 3).mean(0))))
    if kind == "zoom":
        h, w = img.shape[:2]; m = int(round(h * sev / 2))
        return np.asarray(pil.crop((m, m, w - m, h - m)).resize((w, h), Image.BILINEAR))
    if kind == "occlusion":
        h, w = img.shape[:2]; area = sev * h * w
        rh = int(np.sqrt(area)); rw = int(area / rh)
        y0 = rng.integers(0, h - rh); x0 = rng.integers(0, w - rw)
        out = img.copy(); out[y0:y0 + rh, x0:x0 + rw] = img.reshape(-1, 3).mean(0).astype(np.uint8)
        return out
    raise ValueError(kind)


def build_corrupted_sets(x_test: np.ndarray, seed: int = 42):
    out_dir = CACHE / "dimpsar_c"; out_dir.mkdir(exist_ok=True)
    sets = {}
    for kind, sevs in SEVERITIES.items():
        for s_i, sev in enumerate(sevs, 1):
            f = out_dir / f"{kind}_{s_i}.npy"
            if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(x_test):
                sets[(kind, s_i)] = np.load(f); continue
            rng = np.random.default_rng(seed + s_i)
            arr = np.stack([corrupt(x_test[i], kind, sev, rng, i) for i in range(len(x_test))])
            np.save(f, arr); sets[(kind, s_i)] = arr
            print(f"  built {kind} severity {s_i}")
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="*", default=None, help="paths of .keras models (default: all in models/)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    classes, paths, labels, idx = get_split(args.data)
    te = idx["test"][: args.limit] if args.limit else idx["test"]
    x_test = load_images(paths[te], 224, cache_key=None if args.limit else "test")
    y_test = labels[te]
    print("Building DIMPSAR-C ...")
    sets = build_corrupted_sets(x_test)

    model_paths = args.models or sorted(glob.glob("models/*.keras"))
    rows, summary = [], []
    for mp in model_paths:
        name = Path(mp).stem
        model = load_model(mp)  # all saved models take raw [0,255] RGB input
        acc = lambda a: float(np.mean(np.argmax(model.predict(a.astype(np.float32), batch_size=args.batch, verbose=0), 1) == y_test))
        clean = acc(x_test)
        per = {}
        for (kind, s_i), arr in sets.items():
            per[(kind, s_i)] = acc(arr)
            rows.append([name, kind, s_i, f"{per[(kind, s_i)]:.4f}"])
        mca = float(np.mean(list(per.values())))
        worst = min(((k, v) for k, v in per.items() if k[1] == 3), key=lambda kv: kv[1])
        summary.append([name, f"{clean:.4f}", f"{mca:.4f}", f"{mca / clean:.3f}" if clean else "nan",
                        f"{worst[0][0]} ({worst[1]:.3f})"])
        print(f"[{name}] clean={clean:.4f}  mCA={mca:.4f}  rel={mca / max(clean, 1e-9):.3f}  worst={worst[0][0]} {worst[1]:.3f}")

    with open(RESULTS / "robustness.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "corruption", "severity", "accuracy"]); w.writerows(rows)
    with open(RESULTS / "robustness_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "clean_accuracy", "mCA", "relative_robustness", "worst_corruption_sev3"])
        w.writerows(summary)
    print(f"-> {RESULTS / 'robustness.csv'}, {RESULTS / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
