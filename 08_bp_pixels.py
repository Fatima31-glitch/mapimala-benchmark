"""Step 8 - re-run of the BP network of the original MAPIMALA application inside the released pipeline.

BP network = one hidden layer of 256 ReLU units + 40-way softmax on the flattened 64x64x3 image, pixel values
rescaled to [0, 1] (NOT standardised, as in the original application), stochastic gradient descent
(learning rate 0.001, momentum 0.9), batch 32, early stopping on validation accuracy (patience 6, best weights
restored), at most 50 epochs, seed 42, same split as every other model (results/split.json).

Outputs
  results/eval_bp_pixels.json, preds/bp_pixels.npy, models/bp_pixels.keras

Usage (Colab, CPU runtime is enough, ~5-10 min):
  python 08_bp_pixels.py --data "/content/data/Medicinal plant dataset"
"""
import argparse
import time

import numpy as np

from common import MODELS, PREDS, RESULTS, count_params, evaluate, file_size_mb, get_split, latency_ms, load_images, \
    save_json, set_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import keras

    for d in (RESULTS, PREDS, MODELS):
        d.mkdir(exist_ok=True)
    classes, paths, labels, idx = get_split(args.data)
    y = {k: labels[v] for k, v in idx.items()}
    x = {k: load_images(paths[v], 64, cache_key=k).reshape(len(v), -1).astype(np.float32) / 255.0 for k, v in idx.items()}
    print(f"train {x['train'].shape}  val {x['val'].shape}  test {x['test'].shape}  range [{x['train'].min():.2f}, {x['train'].max():.2f}]")

    set_seeds(args.seed)
    model = keras.Sequential([keras.Input((x["train"].shape[1],)),
                              keras.layers.Dense(args.hidden, activation="relu"),
                              keras.layers.Dense(len(classes), activation="softmax")], name="bp_pixels")
    model.compile(optimizer=keras.optimizers.SGD(0.001, momentum=0.9), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    t0 = time.time()
    h = model.fit(x["train"], y["train"], validation_data=(x["val"], y["val"]), epochs=args.epochs, batch_size=32,
                  callbacks=[es], verbose=2)
    t = time.time() - t0

    name = "bp_pixels"
    model.save(MODELS / f"{name}.keras")
    pred = np.argmax(model.predict(x["test"], verbose=0), 1)
    np.save(PREDS / f"{name}.npy", pred)
    lat = latency_ms(lambda a: model.predict(a, verbose=0), x["test"], n=100)
    rec = {"name": name, **evaluate(y["test"], pred, len(classes)), "train_time_s": round(t, 1),
           **count_params(model), "hidden_units": args.hidden, "epochs": len(h.history["loss"]),
           "best_val_accuracy": float(max(h.history["val_accuracy"])), "latency_ms": round(lat, 2),
           "model_size_mb": file_size_mb(MODELS / f"{name}.keras"), "backbone": "none", "regime": "pixels",
           "input": "64x64x3 pixels in [0,1], not standardised", "optimizer": "SGD lr=0.001 momentum=0.9", "seed": args.seed}
    save_json(rec, RESULTS / f"eval_{name}.json")
    print(f"\n[{name}] acc={rec['accuracy']:.4f} [{rec['acc_ci_low']:.3f}-{rec['acc_ci_high']:.3f}]  "
          f"P={rec['precision_macro']:.4f} R={rec['recall_macro']:.4f} F1={rec['f1_macro']:.4f}  "
          f"epochs={rec['epochs']}  train={t:.0f}s  params={rec.get('params_total')}")


if __name__ == "__main__":
    main()
