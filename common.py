"""Shared utilities for the MAPIMALA extended benchmark (DIMPSAR, 40 species).

Everything here is deliberately dependency-light: numpy, scipy, scikit-learn, Pillow, TensorFlow/Keras 3.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
RESULTS = Path("results")
FEATURES = Path("features")
MODELS = Path("models")
PREDS = Path("preds")
FIGURES = Path("figures")
CACHE = Path("cache")
for _d in (RESULTS, FEATURES, MODELS, PREDS, FIGURES, CACHE):
    _d.mkdir(exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# --------------------------------------------------------------------------------------
# Backbones: Keras-applications constructor, preprocessing mode, native input size
# --------------------------------------------------------------------------------------
# mode: "caffe" = RGB->BGR and ImageNet mean subtraction (VGG16, ResNet50)
#       "torch" = /255, ImageNet mean/std (DenseNet121)
#       "none"  = the Keras model already contains its preprocessing (MobileNetV3, EfficientNetV2, ConvNeXt)
BACKBONES = {
    "vgg16":            ("VGG16",            "caffe"),
    "mobilenetv3l":     ("MobileNetV3Large", "none"),
    "efficientnetv2b0": ("EfficientNetV2B0", "none"),
    "densenet121":      ("DenseNet121",      "torch"),
    "resnet50":         ("ResNet50",         "caffe"),
    "convnexttiny":     ("ConvNeXtTiny",     "none"),
}
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_np(x: np.ndarray, mode: str) -> np.ndarray:
    """x: float32 array in [0, 255], RGB, shape (N, H, W, 3)."""
    x = x.astype(np.float32)
    if mode == "caffe":
        return x[..., ::-1] - IMAGENET_MEAN_BGR
    if mode == "torch":
        return (x / 255.0 - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
    return x


def preprocess_tf(x, mode: str):
    """Same as preprocess_np but on TensorFlow tensors (for tf.data pipelines)."""
    import tensorflow as tf
    x = tf.cast(x, tf.float32)
    if mode == "caffe":
        return x[..., ::-1] - tf.constant(IMAGENET_MEAN_BGR)
    if mode == "torch":
        return (x / 255.0 - tf.constant(IMAGENET_MEAN_RGB)) / tf.constant(IMAGENET_STD_RGB)
    return x


def _keras():
    import keras
    return keras


def get_preprocess_layer_class():
    """Serializable Keras layer applying the backbone-specific ImageNet preprocessing to raw [0,255] RGB input.
    Defined lazily (Keras is imported only when needed) and registered once."""
    keras = _keras()
    if hasattr(get_preprocess_layer_class, "_cls"):
        return get_preprocess_layer_class._cls

    @keras.saving.register_keras_serializable(package="mapimala")
    class Preprocess(keras.layers.Layer):
        def __init__(self, mode="none", **kw):
            super().__init__(**kw)
            self.mode = mode

        def call(self, x):
            ops = keras.ops
            x = ops.cast(x, "float32")
            if self.mode == "caffe":
                return ops.flip(x, axis=-1) - ops.convert_to_tensor(IMAGENET_MEAN_BGR)
            if self.mode == "torch":
                return (x / 255.0 - ops.convert_to_tensor(IMAGENET_MEAN_RGB)) / ops.convert_to_tensor(IMAGENET_STD_RGB)
            return x

        def get_config(self):
            return {**super().get_config(), "mode": self.mode}

    get_preprocess_layer_class._cls = Preprocess
    return Preprocess


def make_augmentation(seed: int):
    keras = _keras()
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip("horizontal", seed=seed),
        L.RandomRotation(20 / 360, fill_mode="reflect", seed=seed),
        L.RandomZoom(0.15, fill_mode="reflect", seed=seed),
        L.RandomBrightness(0.2, value_range=(0, 255), seed=seed),
        L.RandomContrast(0.2, seed=seed),
    ], name="augmentation")


def assemble_model(bb, mode: str, n_classes: int, augment: bool = False, seed: int = 42, name: str = "model",
                   head_hidden: int = 256, dropout: float = 0.5):
    """Full deployable model: raw [0,255] RGB 224x224 input -> (augmentation) -> preprocessing -> backbone -> head."""
    keras = _keras()
    Preprocess = get_preprocess_layer_class()
    inp = keras.Input((224, 224, 3), name="image")
    h = inp
    if augment:
        h = make_augmentation(seed)(h)
    h = Preprocess(mode, name="preprocess")(h)
    h = bb(h)
    h = keras.layers.Dense(head_hidden, activation="relu", name="head_dense")(h)
    h = keras.layers.Dropout(dropout, name="head_dropout")(h)
    out = keras.layers.Dense(n_classes, activation="softmax", name="head_softmax")(h)
    return keras.Model(inp, out, name=name)


def load_model(path):
    """Load a saved MAPIMALA model (registers the Preprocess layer first)."""
    keras = _keras()
    get_preprocess_layer_class()
    return keras.models.load_model(path, compile=False)


def build_backbone(key: str, weights: str | None = "imagenet"):
    import keras
    ctor_name, mode = BACKBONES[key]
    ctor = getattr(keras.applications, ctor_name)
    bb = ctor(include_top=False, weights=weights, input_shape=(224, 224, 3), pooling="avg")
    bb.trainable = False
    return bb, mode


# --------------------------------------------------------------------------------------
# Data: listing, stratified split (70/10/20, seed 42), image cache
# --------------------------------------------------------------------------------------
def list_dataset(data_dir: str):
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    paths, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((data_dir / c).iterdir()):
            if f.suffix in IMG_EXT:
                paths.append(str(f))
                labels.append(ci)
    return classes, np.array(paths), np.array(labels, dtype=np.int64)


def get_split(data_dir: str, seed: int = SEED, path: Path = RESULTS / "split.json"):
    """Stratified 70/10/20 split, created once and reused by every script (released with the paper)."""
    classes, paths, labels = list_dataset(data_dir)
    if path.exists():
        split = json.loads(path.read_text())
        assert split["classes"] == classes, "Class list changed since the split was created"
        idx = {k: np.array(v) for k, v in split["indices"].items()}
        # sanity: the split was built on the same file list
        assert len(split["paths"]) == len(paths) and split["paths"][0] == os.path.relpath(paths[0], data_dir)
        return classes, paths, labels, idx
    from sklearn.model_selection import train_test_split
    all_idx = np.arange(len(paths))
    tr, rest = train_test_split(all_idx, test_size=0.30, random_state=seed, stratify=labels)
    va, te = train_test_split(rest, test_size=2 / 3, random_state=seed, stratify=labels[rest])
    idx = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
    path.write_text(json.dumps({
        "seed": seed, "classes": classes,
        "paths": [os.path.relpath(p, data_dir) for p in paths],
        "indices": {k: v.tolist() for k, v in idx.items()},
    }, indent=1))
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} -> {path}")
    return classes, paths, labels, idx


def load_images(paths, size: int, cache_key: str | None = None) -> np.ndarray:
    """Load and resize images to (size, size), RGB, uint8. Cached as .npy to make re-runs fast."""
    if cache_key is not None:
        f = CACHE / f"{cache_key}_{size}.npy"
        if f.exists():
            return np.load(f)
    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)")
    if cache_key is not None:
        np.save(CACHE / f"{cache_key}_{size}.npy", out)
    return out


# --------------------------------------------------------------------------------------
# Metrics and statistics
# --------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def evaluate(y_true, y_pred, n_classes: int, n_boot: int = 2000, seed: int = SEED) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    lo, hi = wilson(int((y_true == y_pred).sum()), len(y_true))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(y_true), len(y_true))
        boots.append(f1_score(y_true[ii], y_pred[ii], average="macro", zero_division=0))
    return {
        "n_test": int(len(y_true)), "accuracy": float(acc), "acc_ci_low": float(lo), "acc_ci_high": float(hi),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "f1_ci_low": float(np.percentile(boots, 2.5)), "f1_ci_high": float(np.percentile(boots, 97.5)),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on discordant pairs. correct_*: boolean arrays on the same test images."""
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else binomtest(b, n, 0.5).pvalue
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def per_class_table(y_true, y_pred, classes) -> list[dict]:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    return [{"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)]


def confused_pairs(y_true, y_pred, classes, top: int = 10) -> list[dict]:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    flat = [(int(cm[i, j]), i, j) for i in range(len(classes)) for j in range(len(classes)) if cm[i, j] > 0]
    flat.sort(reverse=True)
    return [{"true": classes[i], "predicted": classes[j], "count": n} for n, i, j in flat[:top]]


def count_params(model) -> dict:
    import keras
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    return {"params_total": total, "params_trainable": trainable}


def latency_ms(predict_fn, x: np.ndarray, n: int = 200) -> float:
    """Median per-image latency (batch size 1) in milliseconds, after 5 warm-up calls."""
    n = min(n, len(x))
    for i in range(5):
        predict_fn(x[i:i + 1])
    times = []
    for i in range(n):
        t0 = time.perf_counter(); predict_fn(x[i:i + 1]); times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def file_size_mb(path) -> float:
    return round(os.path.getsize(path) / 1e6, 2)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))


def load_json(path):
    return json.loads(Path(path).read_text())


def set_seeds(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except Exception:
        pass
