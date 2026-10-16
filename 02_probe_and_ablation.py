"""Step 2 - (a) frozen-backbone heads, (b) the from-scratch CNN, (c) the representation x classifier ablation.

(a) For every backbone in features/, train the MAPIMALA head (Dense 256 ReLU + Dropout + softmax) on the cached
    features, evaluate on the test set, save predictions and a full deployable model (backbone + head).
(b) Train the from-scratch CNN of the paper (4 conv blocks 32/64/128/128, Dense 256, dropout) and cache its
    penultimate (pooled) features for the ablation.
(c) Representation x classifier grid: {logistic regression, BP (1 hidden), DNN (2 hidden), GA-BP} x
    {raw pixels 64x64, PCA-100, CNN-scratch features, each backbone's features}.

Outputs
  preds/<name>.npy, results/eval_<name>.json, models/<name>.keras, results/ablation_grid.csv
"""
import argparse
import csv
import time

import numpy as np

from common import (BACKBONES, FEATURES, MODELS, PREDS, RESULTS, assemble_model, build_backbone, count_params, evaluate,
                    get_split, latency_ms, load_images, save_json, set_seeds, file_size_mb)

N_CLASSES = 40


# ----------------------------------------------------------------------------- heads / classifiers
def make_head(input_dim, n_classes, hidden=256, dropout=0.5):
    import keras
    return keras.Sequential([keras.Input((input_dim,)), keras.layers.Dense(hidden, activation="relu"),
                             keras.layers.Dropout(dropout), keras.layers.Dense(n_classes, activation="softmax")])


def make_mlp(input_dim, n_classes, hidden=(256,), dropout=0.0):
    import keras
    layers = [keras.Input((input_dim,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu"))
        if dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))
    return keras.Sequential(layers)


def fit(model, xtr, ytr, xva, yva, epochs, lr=1e-3, optimizer="adam", patience=6, batch=32, verbose=0):
    import keras
    opt = keras.optimizers.Adam(lr) if optimizer == "adam" else keras.optimizers.SGD(lr, momentum=0.9)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)
    t0 = time.time()
    h = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=epochs, batch_size=batch, callbacks=[es], verbose=verbose)
    return time.time() - t0, len(h.history["loss"])


# ----------------------------------------------------------------------------- GA-BP (numpy GA + Keras refinement)
def ga_init_weights(xtr, ytr, xva, yva, n_in, n_hidden=32, n_out=N_CLASSES, pop=50, gens=50, tourn=3, pc=0.7,
                    pm=0.15, elite=2, seed=42, verbose=True):
    """Genetic algorithm over the flat weight vector of a 1-hidden-layer network; fitness = validation accuracy."""
    rng = np.random.default_rng(seed)
    n_w = n_in * n_hidden + n_hidden + n_hidden * n_out + n_out

    def unpack(v):
        i = 0
        w1 = v[i:i + n_in * n_hidden].reshape(n_in, n_hidden); i += n_in * n_hidden
        b1 = v[i:i + n_hidden]; i += n_hidden
        w2 = v[i:i + n_hidden * n_out].reshape(n_hidden, n_out); i += n_hidden * n_out
        b2 = v[i:i + n_out]
        return w1, b1, w2, b2

    def fitness(v):
        w1, b1, w2, b2 = unpack(v)
        h = np.maximum(xva @ w1 + b1, 0)
        return float(np.mean(np.argmax(h @ w2 + b2, 1) == yva))

    popu = rng.normal(0, 0.1, (pop, n_w))
    fits = np.array([fitness(v) for v in popu])
    for g in range(gens):
        order = np.argsort(-fits)
        new = [popu[i].copy() for i in order[:elite]]
        while len(new) < pop:
            def pick():
                c = rng.integers(0, pop, tourn); return popu[c[np.argmax(fits[c])]]
            a, b = pick().copy(), pick().copy()
            if rng.random() < pc:  # uniform crossover
                m = rng.random(n_w) < 0.5
                a[m], b[m] = b[m], a[m]
            for child in (a, b):
                m = rng.random(n_w) < pm
                child[m] += rng.normal(0, 0.1, m.sum())
                if len(new) < pop:
                    new.append(child)
        popu = np.array(new)
        fits = np.array([fitness(v) for v in popu])
        if verbose and (g % 10 == 0 or g == gens - 1):
            print(f"    GA gen {g:3d}: best val acc = {fits.max():.4f}")
    return unpack(popu[np.argmax(fits)])


def gabp(xtr, ytr, xva, yva, n_classes, n_hidden=32, epochs=120, seed=42, gens=50):
    import keras
    w1, b1, w2, b2 = ga_init_weights(xtr, ytr, xva, yva, xtr.shape[1], n_hidden, n_out=n_classes, gens=gens, seed=seed)
    model = make_mlp(xtr.shape[1], n_classes, hidden=(n_hidden,))
    model.layers[0].set_weights([w1, b1]); model.layers[1].set_weights([w2, b2])
    t, ep = fit(model, xtr, ytr, xva, yva, epochs=epochs, lr=1e-3)
    return model, t


# ----------------------------------------------------------------------------- from-scratch CNN
def make_scratch_cnn(n_classes=N_CLASSES, filters=(32, 64, 128, 128), dense=256, dropout=0.5):
    import keras
    L = keras.layers
    layers = [keras.Input((224, 224, 3)), L.Rescaling(1.0 / 255)]
    for f in filters:
        layers += [L.Conv2D(f, 3, padding="same", activation="relu"), L.MaxPooling2D(2)]
    layers += [L.Flatten(), L.Dense(dense, activation="relu", name="penultimate"), L.Dropout(dropout),
               L.Dense(n_classes, activation="softmax")]
    return keras.Sequential(layers, name="scratch_cnn")


def record(name, model_or_fn, xte_eval, yte, classes, train_time, extra=None, latency_input=None):
    """Evaluate, save predictions and a JSON record with cost figures."""
    pred = np.argmax(model_or_fn.predict(xte_eval, verbose=0), 1) if hasattr(model_or_fn, "predict") else model_or_fn(xte_eval)
    np.save(PREDS / f"{name}.npy", pred)
    rec = {"name": name, **evaluate(yte, pred, len(classes)), "train_time_s": round(train_time, 1)}
    if extra:
        rec.update(extra)
    save_json(rec, RESULTS / f"eval_{name}.json")
    print(f"  [{name}] acc={rec['accuracy']:.4f} [{rec['acc_ci_low']:.3f}-{rec['acc_ci_high']:.3f}] "
          f"F1={rec['f1_macro']:.4f}  train={train_time:.0f}s")
    return pred, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiply all epoch budgets (smoke tests: 0.05)")
    ap.add_argument("--skip-scratch", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--ga-gens", type=int, default=50)
    args = ap.parse_args()
    import keras
    set_seeds(42)
    weights = None if args.weights == "none" else args.weights
    E = lambda n: max(1, int(round(n * args.epochs_scale)))

    classes, paths, labels, idx = get_split(args.data)
    if args.limit:
        idx = {k: v[: args.limit] for k, v in idx.items()}
    y = {k: labels[v] for k, v in idx.items()}
    x224 = {k: load_images(paths[v], 224, cache_key=None if args.limit else k) for k, v in idx.items()}
    x64 = {k: load_images(paths[v], 64, cache_key=None if args.limit else k) for k, v in idx.items()}

    # ---------------- (a) frozen heads on cached features + deployable models
    backbone_keys = [k.strip() for k in args.backbones.split(",")]
    feats = {}
    for key in backbone_keys:
        f = np.load(FEATURES / f"{key}.npz")
        feats[key] = {s: f[f"x_{s}"] for s in ("train", "val", "test")}
        if args.limit:
            feats[key] = {s: v[: args.limit] for s, v in feats[key].items()}
        print(f"\n=== frozen head on {key} ===")
        set_seeds(42)
        head = make_head(feats[key]["train"].shape[1], len(classes))
        t, ep = fit(head, feats[key]["train"], y["train"], feats[key]["val"], y["val"], epochs=E(40))
        # deployable model = preprocessing + frozen backbone + trained head; raw [0,255] RGB input
        bb, mode = build_backbone(key, weights)
        full = assemble_model(bb, mode, len(classes), augment=False, name=f"{key}_frozen")
        full.get_layer("head_dense").set_weights(head.layers[0].get_weights())
        full.get_layer("head_softmax").set_weights(head.layers[2].get_weights())
        full.save(MODELS / f"{key}_frozen.keras")
        lat = latency_ms(lambda a: full.predict(a.astype(np.float32), verbose=0), x224["test"], n=100)
        extra = {**count_params(full), "head_params": int(head.count_params()), "epochs": ep, "latency_ms": round(lat, 2),
                 "model_size_mb": file_size_mb(MODELS / f"{key}_frozen.keras"), "backbone": key, "regime": "frozen"}
        record(f"{key}_frozen", head, feats[key]["test"], y["test"], classes, t, extra)

    # ---------------- (b) from-scratch CNN (also provides learned features for the ablation)
    cnn_feats = None
    if not args.skip_scratch:
        print("\n=== CNN from scratch ===")
        set_seeds(42)
        cnn = make_scratch_cnn(len(classes))
        t, ep = fit(cnn, x224["train"], y["train"], x224["val"], y["val"], epochs=E(30))
        cnn.save(MODELS / "cnn_scratch.keras")
        lat = latency_ms(lambda a: cnn.predict(a.astype(np.float32), verbose=0), x224["test"], n=100)
        extra = {**count_params(cnn), "epochs": ep, "latency_ms": round(lat, 2),
                 "model_size_mb": file_size_mb(MODELS / "cnn_scratch.keras"), "backbone": "none", "regime": "scratch"}
        record("cnn_scratch", cnn, x224["test"].astype(np.float32), y["test"], classes, t, extra)
        fe = keras.Model(cnn.inputs, cnn.get_layer("penultimate").output)
        cnn_feats = {s: fe.predict(x224[s].astype(np.float32), batch_size=32, verbose=0) for s in ("train", "val", "test")}
        np.savez_compressed(FEATURES / "cnn_scratch.npz", **{f"x_{k}": v for k, v in cnn_feats.items()})

    # ---------------- (c) representation x classifier ablation
    if args.skip_ablation:
        return
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("\n=== representation x classifier ablation ===")
    reps = {}
    px = {s: x64[s].reshape(len(x64[s]), -1).astype(np.float32) / 255.0 for s in x64}
    reps["raw_pixels_64"] = px
    n_pc = min(100, len(px["train"]) - 1, px["train"].shape[1])
    pca = PCA(n_pc, random_state=42).fit(px["train"])
    sc = StandardScaler().fit(pca.transform(px["train"]))
    reps["pca100"] = {s: sc.transform(pca.transform(px[s])).astype(np.float32) for s in px}
    if cnn_feats is not None:
        reps["cnn_scratch_features"] = cnn_feats
    for key in backbone_keys:
        reps[f"{key}_features"] = feats[key]

    rows = []
    for rname, r in reps.items():
        scaler = StandardScaler().fit(r["train"])
        rs = {s: scaler.transform(r[s]).astype(np.float32) for s in r}
        print(f"  --- representation: {rname} (dim {rs['train'].shape[1]})")
        # logistic regression
        t0 = time.time()
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(rs["train"], y["train"])
        pred = lr.predict(rs["test"]); np.save(PREDS / f"abl_logreg_{rname}.npy", pred)
        rows.append(["logreg", rname, evaluate(y["test"], pred, len(classes), n_boot=200)["accuracy"], time.time() - t0])
        # BP (1 hidden layer, SGD+momentum) and DNN (512-256, Adam)
        for cname, hidden, opt, ep in (("bp", (256,), "sgd", 50), ("dnn", (512, 256), "adam", 40)):
            set_seeds(42)
            m = make_mlp(rs["train"].shape[1], len(classes), hidden=hidden)
            t, _ = fit(m, rs["train"], y["train"], rs["val"], y["val"], epochs=E(ep), optimizer=opt)
            pred = np.argmax(m.predict(rs["test"], verbose=0), 1); np.save(PREDS / f"abl_{cname}_{rname}.npy", pred)
            rows.append([cname, rname, evaluate(y["test"], pred, len(classes), n_boot=200)["accuracy"], t])
        # GA-BP on <= 100 dims only (as in the paper: PCA-100); for high-dim features use PCA-100 of them
        if rs["train"].shape[1] > 100:
            p2 = PCA(min(100, len(rs["train"]) - 1), random_state=42).fit(rs["train"]); rg = {s: p2.transform(rs[s]).astype(np.float32) for s in rs}
        else:
            rg = rs
        t0 = time.time()
        m, t = gabp(rg["train"], y["train"], rg["val"], y["val"], len(classes), epochs=E(120), gens=args.ga_gens)
        pred = np.argmax(m.predict(rg["test"], verbose=0), 1); np.save(PREDS / f"abl_gabp_{rname}.npy", pred)
        rows.append(["gabp", rname, evaluate(y["test"], pred, len(classes), n_boot=200)["accuracy"], time.time() - t0])
        for r_ in rows[-4:]:
            print(f"    {r_[0]:7s} acc={r_[2]:.4f}  ({r_[3]:.0f}s)")

    with open(RESULTS / "ablation_grid.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["classifier", "representation", "test_accuracy", "train_time_s"])
        for r_ in rows:
            w.writerow([r_[0], r_[1], f"{r_[2]:.4f}", f"{r_[3]:.1f}"])
    print(f"-> {RESULTS / 'ablation_grid.csv'}")


if __name__ == "__main__":
    main()
