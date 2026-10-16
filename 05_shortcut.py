"""Step 5 - Shortcut analysis: Grad-CAM attribution mass on the leaf, masked-image accuracies, background-only classifier.

(i)  Leaf mask: HSV threshold (green/yellow hue, saturation) + morphological opening + hole filling.
     Optional GrabCut fallback if OpenCV is installed (pip install opencv-python-headless).
(ii) Grad-CAM (last 4-D layer of the backbone) -> CAM-in-leaf ratio for N test images per convolutional model.
(iii) Accuracy on leaf-only images (background -> train mean colour) and background-only images (leaf -> mean colour).
(iv) Background-only classifier: frozen-feature head trained on background-only training images.

Outputs
  results/shortcut.csv, results/shortcut_background_classifier.json, figures/gradcam_panel.pdf, cache/leaf_masks_test.npy

Usage
  python 05_shortcut.py --data DIR --models models/vgg16_frozen.keras models/cnn_scratch.keras ... [--n-cam 200]
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from common import (CACHE, FIGURES, RESULTS, build_backbone, evaluate, get_split, load_images, load_model, preprocess_np,
                    save_json, set_seeds)


# ----------------------------------------------------------------------------- leaf masks
def leaf_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the leaf region (True = leaf). Vegetation hue in PIL's HSV: ~ 35-110 (of 255)."""
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 25) & (h <= 115) & (s >= 50) & (v >= 30)
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    # keep the largest connected component (+ anything above 10 % of its size)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, sz in enumerate(sizes) if sz >= 0.1 * sizes.max()]
        m = np.isin(lab, keep)
    frac = m.mean()
    if frac < 0.05 or frac > 0.95:
        m = grabcut_mask(img, m)
    return m


def grabcut_mask(img, fallback):
    try:
        import cv2
    except ImportError:
        return fallback
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return m if 0.05 <= m.mean() <= 0.95 else fallback
    except Exception:
        return fallback


def masks_for(x: np.ndarray, cache_key: str):
    f = CACHE / f"leaf_masks_{cache_key}.npy"
    if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(x):
        return np.load(f)
    m = np.stack([leaf_mask(im) for im in x])
    np.save(f, m)
    return m


# ----------------------------------------------------------------------------- Grad-CAM
def _has_conv(m):
    import keras
    return any(isinstance(l, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.SeparableConv2D)) for l in m.layers)


def split_model(model):
    """Return (pre_layers, backbone_or_None, conv_layer_name, post_layers) for a saved MAPIMALA model."""
    import keras
    layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    subs = [l for l in layers if isinstance(l, keras.Model) and _has_conv(l)]
    if subs:
        bb = subs[0]
        k = layers.index(bb)
        conv = [l for l in bb.layers if hasattr(l, "output") and len(l.output.shape) == 4][-1]
        return layers[:k], bb, conv.name, layers[k + 1:]
    conv = [l for l in layers if hasattr(l, "output") and len(l.output.shape) == 4][-1]
    return [], None, conv.name, []


def gradcam_fn(model):
    """Build a function image(1,H,W,3 raw) -> (cam HxW in [0,1], predicted class)."""
    import keras
    import tensorflow as tf
    pre, bb, conv_name, post = split_model(model)
    if bb is not None:
        core = keras.Model(bb.input, [bb.get_layer(conv_name).output, bb.output])
    else:  # flat Sequential (scratch CNN): rebuild functionally, tapping the last 4-D layer
        inp = keras.Input(model.layers[0].input.shape[1:] if hasattr(model.layers[0], "input") else (224, 224, 3))
        h, tap = inp, None
        for l in model.layers:
            if isinstance(l, keras.layers.InputLayer):
                continue
            h = l(h)
            if l.name == conv_name:
                tap = h
        core = keras.Model(inp, [tap, h])

    def run(x_raw):
        x = tf.convert_to_tensor(x_raw.astype(np.float32))
        for l in pre:
            x = l(x, training=False)
        with tf.GradientTape() as tape:
            conv, h = core(x, training=False)
            for l in post:
                h = l(h, training=False)
            c = tf.argmax(h[0])
            score = h[:, c]
        grads = tape.gradient(score, conv)
        w = tf.reduce_mean(grads, axis=(1, 2))[0]
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
        cam = ndimage.zoom(cam, (x_raw.shape[1] / cam.shape[0], x_raw.shape[2] / cam.shape[1]), order=1)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, int(c)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-cam", type=int, default=200, help="test images used for the CAM-in-leaf ratio")
    ap.add_argument("--bg-backbone", default="mobilenetv3l", help="backbone for the background-only classifier")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    import keras
    set_seeds(42)
    weights = None if args.weights == "none" else args.weights

    classes, paths, labels, idx = get_split(args.data)
    if args.limit:
        idx = {k: v[: args.limit] for k, v in idx.items()}
    y = {k: labels[v] for k, v in idx.items()}
    x = {k: load_images(paths[v], 224, cache_key=None if args.limit else k) for k, v in idx.items()}
    print("Computing leaf masks ...")
    masks = {k: masks_for(x[k], k if not args.limit else f"{k}_lim{args.limit}") for k in x}
    leaf_frac = float(masks["test"].mean())
    mean_col = x["train"].reshape(-1, 3).mean(0).astype(np.uint8)
    print(f"  leaf-area fraction (test) = {leaf_frac:.3f}")

    def masked(xs, ms, keep_leaf: bool):
        out = xs.copy()
        sel = ~ms if keep_leaf else ms
        out[sel] = mean_col
        return out

    x_leaf = {k: masked(x[k], masks[k], True) for k in x}
    x_bg = {k: masked(x[k], masks[k], False) for k in x}

    rows = []
    model_paths = args.models or sorted(glob.glob("models/*.keras"))
    panel = {}
    rng = np.random.default_rng(42)
    cam_idx = rng.choice(len(x["test"]), min(args.n_cam, len(x["test"])), replace=False)
    for mp in model_paths:
        name = Path(mp).stem
        model = load_model(mp)  # raw [0,255] input
        acc = lambda a: float(np.mean(np.argmax(model.predict(a.astype(np.float32), batch_size=32, verbose=0), 1) == y["test"]))
        clean, a_leaf, a_bg = acc(x["test"]), acc(x_leaf["test"]), acc(x_bg["test"])
        # Grad-CAM
        ratios = []
        try:
            run_cam = gradcam_fn(model)
            for i in cam_idx:
                cam, _ = run_cam(x["test"][i:i + 1])
                m = masks["test"][i]
                ratios.append(float(cam[m].sum() / cam.sum()) if cam.sum() > 0 else np.nan)
                if len(panel.get(name, [])) < 6:
                    panel.setdefault(name, []).append((i, cam))
            cam_ratio = float(np.nanmean(ratios))
        except Exception as e:  # pragma: no cover
            print(f"  Grad-CAM failed for {name}: {e}")
            cam_ratio = float("nan")
        rows.append([name, f"{cam_ratio:.3f}", f"{clean:.4f}", f"{a_leaf:.4f}", f"{a_bg:.4f}"])
        print(f"[{name}] CAM-in-leaf={cam_ratio:.3f} (leaf area {leaf_frac:.3f})  clean={clean:.4f}  "
              f"leaf-only={a_leaf:.4f}  background-only={a_bg:.4f}")

    # background-only classifier (Noyan-style): frozen features of background-only images
    print(f"\nBackground-only classifier on {args.bg_backbone} features ...")
    bb, mode = build_backbone(args.bg_backbone, weights)
    f = {k: bb.predict(preprocess_np(x_bg[k], mode), batch_size=32, verbose=0) for k in x_bg}
    head = keras.Sequential([keras.Input((f["train"].shape[1],)), keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dropout(0.5), keras.layers.Dense(len(classes), activation="softmax")])
    head.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(f["train"], y["train"], validation_data=(f["val"], y["val"]), epochs=args.epochs, batch_size=32, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)])
    pred = np.argmax(head.predict(f["test"], verbose=0), 1)
    bg_eval = evaluate(y["test"], pred, len(classes), n_boot=500)
    bg_eval.update({"chance": 1 / len(classes), "backbone": args.bg_backbone, "leaf_area_fraction_test": leaf_frac})
    save_json(bg_eval, RESULTS / "shortcut_background_classifier.json")
    print(f"  background-only accuracy = {bg_eval['accuracy']:.4f} [{bg_eval['acc_ci_low']:.3f}-{bg_eval['acc_ci_high']:.3f}]"
          f"  (chance {1 / len(classes):.3f})")
    rows.append(["background_only_classifier_" + args.bg_backbone, "", "", "", f"{bg_eval['accuracy']:.4f}"])

    with open(RESULTS / "shortcut.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "cam_in_leaf_ratio", "clean_accuracy", "accuracy_leaf_only", "accuracy_background_only"])
        w.writerow(["leaf_area_fraction_test", f"{leaf_frac:.3f}", "", "", ""])
        w.writerows(rows)
    print(f"-> {RESULTS / 'shortcut.csv'}")

    # Grad-CAM panel
    if panel:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(panel)
        ncol = min(6, max(len(v) for v in panel.values()))
        fig, axes = plt.subplots(len(names), ncol, figsize=(2.2 * ncol, 2.3 * len(names)), squeeze=False)
        for r, name in enumerate(names):
            for c in range(ncol):
                ax = axes[r, c]; ax.axis("off")
                if c < len(panel[name]):
                    i, cam = panel[name][c]
                    ax.imshow(x["test"][i]); ax.imshow(cam, cmap="jet", alpha=0.45)
                    ax.contour(masks["test"][i], levels=[0.5], colors="white", linewidths=0.8)
                    ratio = cam[masks["test"][i]].sum() / cam.sum() if cam.sum() > 0 else float("nan")
                    ax.set_title(f"{classes[y['test'][i]]}\nCAM-in-leaf {ratio:.2f}", fontsize=7)
            axes[r, 0].text(-0.15, 0.5, name, transform=axes[r, 0].transAxes, rotation=90, va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "gradcam_panel.pdf"); fig.savefig(FIGURES / "gradcam_panel.png", dpi=200)
        print(f"-> {FIGURES / 'gradcam_panel.pdf'}")


if __name__ == "__main__":
    main()
