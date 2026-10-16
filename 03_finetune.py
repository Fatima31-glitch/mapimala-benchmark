"""Step 3 - Fine-tune ImageNet backbones with augmentation, several seeds, on the fixed split.

Two stages per run: (1) head only (backbone frozen), (2) top fraction of the backbone unfrozen with lr/10.
Augmentation (training images only, applied inside the model so it is inactive at inference): horizontal flip,
rotation +/-20 deg, zoom +/-15 %, brightness +/-20 %, contrast +/-20 %. BatchNorm layers stay frozen during stage 2.
Saved models take raw [0,255] RGB 224x224 input (preprocessing is a layer of the model).

Outputs
  models/<backbone>_ft_seed<s>.keras, preds/<backbone>_ft_seed<s>.npy, results/eval_<backbone>_ft_seed<s>.json

Usage (CPU: start with the light backbones)
  python 03_finetune.py --data DIR --backbones mobilenetv3l,efficientnetv2b0,densenet121 --seeds 42,43,44
  Budget knobs: --epochs1 (head) --epochs2 (fine-tune) --unfreeze 0.3 (fraction of backbone layers) --limit N
"""
import argparse
import time

import numpy as np

from common import (BACKBONES, MODELS, PREDS, RESULTS, assemble_model, build_backbone, count_params, evaluate,
                    file_size_mb, get_split, latency_ms, load_images, save_json, set_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbones", default="mobilenetv3l,efficientnetv2b0,densenet121")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--epochs1", type=int, default=10, help="stage 1: head only")
    ap.add_argument("--epochs2", type=int, default=20, help="stage 2: partial fine-tuning (early stopping, patience 4)")
    ap.add_argument("--unfreeze", type=float, default=0.3, help="fraction of backbone layers (from the top) to unfreeze")
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import keras
    weights = None if args.weights == "none" else args.weights

    classes, paths, labels, idx = get_split(args.data)
    if args.limit:
        idx = {k: v[: args.limit] for k, v in idx.items()}
    y = {k: labels[v] for k, v in idx.items()}
    x = {k: load_images(paths[v], 224, cache_key=None if args.limit else k) for k, v in idx.items()}

    for key in [k.strip() for k in args.backbones.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            # "_noaug" keeps the no-augmentation ablation from overwriting the main runs
            name = f"{key}_ft{'_noaug' if args.no_augment else ''}_seed{seed}"
            print(f"\n=== {name} ===")
            set_seeds(seed)
            bb, mode = build_backbone(key, weights)
            model = assemble_model(bb, mode, len(classes), augment=not args.no_augment, seed=seed, name=name)
            xtr, xva = x["train"].astype(np.float32), x["val"].astype(np.float32)
            es = lambda p: keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=p, restore_best_weights=True)

            # stage 1: head only (augmentation layers are active only during training)
            t0 = time.time()
            model.compile(optimizer=keras.optimizers.Adam(args.lr1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            h1 = model.fit(xtr, y["train"], validation_data=(xva, y["val"]), epochs=args.epochs1, batch_size=args.batch,
                           callbacks=[es(4)], verbose=2)

            # stage 2: unfreeze the top fraction of the backbone (BatchNorm kept frozen)
            n_layers = len(bb.layers)
            start = int(n_layers * (1 - args.unfreeze))
            bb.trainable = True
            for i, layer in enumerate(bb.layers):
                layer.trainable = (i >= start) and not isinstance(layer, keras.layers.BatchNormalization)
            model.compile(optimizer=keras.optimizers.Adam(args.lr2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            h2 = model.fit(xtr, y["train"], validation_data=(xva, y["val"]), epochs=args.epochs2, batch_size=args.batch,
                           callbacks=[es(4)], verbose=2)
            train_time = time.time() - t0

            # evaluation + cost (raw [0,255] input; preprocessing is inside the model)
            xte = x["test"].astype(np.float32)
            pred = np.argmax(model.predict(xte, batch_size=args.batch, verbose=0), 1)
            np.save(PREDS / f"{name}.npy", pred)
            model.save(MODELS / f"{name}.keras")
            lat = latency_ms(lambda a: model.predict(a.astype(np.float32), verbose=0), x["test"], n=100)
            rec = {"name": name, "backbone": key, "regime": "finetuned", "seed": seed,
                   **evaluate(y["test"], pred, len(classes)), **count_params(model),
                   "epochs_stage1": len(h1.history["loss"]), "epochs_stage2": len(h2.history["loss"]),
                   "unfreeze_fraction": args.unfreeze, "augment": not args.no_augment,
                   "train_time_s": round(train_time, 1), "latency_ms": round(lat, 2),
                   "model_size_mb": file_size_mb(MODELS / f"{name}.keras"),
                   "best_val_acc": float(max(h1.history["val_accuracy"] + h2.history["val_accuracy"]))}
            save_json(rec, RESULTS / f"eval_{name}.json")
            print(f"  acc={rec['accuracy']:.4f} [{rec['acc_ci_low']:.3f}-{rec['acc_ci_high']:.3f}] F1={rec['f1_macro']:.4f} "
                  f"train={train_time / 60:.1f} min  latency={lat:.1f} ms")


if __name__ == "__main__":
    main()
