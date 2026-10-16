"""Step 6 - Aggregate everything into the tables and figures referenced by the manuscript.

Outputs (results/)
  summary_main.csv      -> Table 3 (accuracy, Wilson CI, macro P/R/F1; fine-tuned models as mean +/- sd over seeds)
  cost.csv              -> Table 7 (parameters, training time, latency, size, accuracy)
  mcnemar.csv           -> Table 4 (pairwise exact McNemar, Holm-corrected)
  per_class.csv         -> Table 5 (per-class P/R/F1 of the best model)
  confused_pairs.csv    -> Table 5 (ten most frequent confusions of the best model)
  summary.md            -> everything above as Markdown, ready to paste
figures/pareto.pdf, figures/robustness_curves.pdf
"""
import csv
import glob
import pathlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import FIGURES, PREDS, RESULTS, confused_pairs, get_split, holm, load_json, mcnemar_exact, per_class_table


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    classes, paths, labels, idx = get_split(args.data)
    y_test = labels[idx["test"]]

    evals = [load_json(f) for f in sorted(glob.glob(str(RESULTS / "eval_*.json")))]
    if not evals:
        raise SystemExit("no results/eval_*.json found - run scripts 02/03 first")

    # ---- group fine-tuned seeds
    groups = defaultdict(list)
    for e in evals:
        base = e["name"].split("_seed")[0]
        groups[base].append(e)

    def fmt(v, d=4): return f"{v:.{d}f}"
    main_rows, cost_rows = [], []
    best = None
    for base, es in groups.items():
        accs = np.array([e["accuracy"] for e in es]); f1s = np.array([e["f1_macro"] for e in es])
        rep = sorted(es, key=lambda e: e["accuracy"])[len(es) // 2]  # median-seed model
        if len(es) > 1:
            acc_s = f"{accs.mean():.4f} ± {accs.std(ddof=1):.4f}"; f1_s = f"{f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}"
        else:
            acc_s, f1_s = fmt(accs[0]), fmt(f1s[0])
        main_rows.append([base, acc_s, f"{rep['acc_ci_low']:.3f}–{rep['acc_ci_high']:.3f}", fmt(rep["precision_macro"]),
                          fmt(rep["recall_macro"]), f1_s, f"{rep['f1_ci_low']:.3f}–{rep['f1_ci_high']:.3f}", len(es), rep["name"]])
        cost_rows.append([base, f"{rep.get('params_total', 0):,} / {rep.get('params_trainable', 0):,}",
                          f"{np.mean([e['train_time_s'] for e in es]) / 60:.1f}", f"{rep.get('latency_ms', float('nan')):.1f}",
                          f"{rep.get('model_size_mb', float('nan'))}", f"{100 * accs.mean():.2f}"])
        # ablation runs (no augmentation) are reported but never selected as the best model
        if "_noaug" not in base and (best is None or accs.mean() > best[1]):
            best = (rep["name"], accs.mean())
    main_rows.sort(key=lambda r: -float(r[1].split(" ")[0]))

    with open(RESULTS / "summary_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "acc_95ci_wilson", "precision_macro", "recall_macro", "f1_macro", "f1_95ci_bootstrap", "n_seeds", "representative_run"])
        w.writerows(main_rows)
    with open(RESULTS / "cost.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "params_total/trainable", "train_time_min", "latency_ms_per_image", "model_size_mb", "accuracy_pct"]); w.writerows(cost_rows)

    # ---- McNemar between representative runs (all pairs, Holm)
    preds = {r[8]: np.load(PREDS / f"{r[8]}.npy") for r in main_rows if (PREDS / f"{r[8]}.npy").exists()}
    names = list(preds); pairs, pv = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca, cb = preds[names[i]] == y_test, preds[names[j]] == y_test
            t = mcnemar_exact(ca, cb); pairs.append((names[i], names[j], t)); pv.append(t["p_value"])
    adj = holm(pv) if pv else []
    with open(RESULTS / "mcnemar.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_a", "model_b", "a_right_b_wrong", "a_wrong_b_right", "p_exact", "p_holm"])
        for (a, b, t), p in zip(pairs, adj):
            w.writerow([a, b, t["a_right_b_wrong"], t["a_wrong_b_right"], f"{t['p_value']:.2e}", f"{p:.2e}"])

    # ---- per-class and confusions of the best model (+ a compact dump of every prediction vector for offline analysis)
    with open(RESULTS / "predictions_test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + [f"img{i}" for i in range(len(y_test))])
        w.writerow(["TRUE"] + list(map(int, y_test)))
        for pf in sorted(glob.glob(str(PREDS / "*.npy"))):
            w.writerow([pathlib.Path(pf).stem] + list(map(int, np.load(pf))))
    bp = preds[best[0]]
    pc = per_class_table(y_test, bp, classes); cp = confused_pairs(y_test, bp, classes, top=10)
    with open(RESULTS / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pc[0].keys()); w.writeheader(); w.writerows(pc)
    with open(RESULTS / "confused_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cp[0].keys()); w.writeheader(); w.writerows(cp)

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = []
    for r in cost_rows:
        try:
            lat, acc, params = float(r[3]), float(r[5]), int(r[1].split(" / ")[0].replace(",", ""))
        except ValueError:
            continue
        pts.append((lat, acc, params, r[0]))
        ax.scatter(lat, acc, s=20 + params / 2e5, alpha=0.7); ax.annotate(r[0], (lat, acc), fontsize=6, xytext=(3, 3), textcoords="offset points")
    pts.sort()
    front, best_acc = [], -1
    for lat, acc, _, _ in pts:
        if acc > best_acc:
            front.append((lat, acc)); best_acc = acc
    if front:
        ax.plot(*zip(*front), "k--", lw=0.8, label="Pareto front")
    ax.set_xscale("log"); ax.set_xlabel("CPU inference latency (ms / image, batch 1)"); ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(FIGURES / "pareto.pdf"); fig.savefig(FIGURES / "pareto.png", dpi=200)

    rob = RESULTS / "robustness.csv"
    if rob.exists():
        rows = list(csv.DictReader(open(rob)))
        kinds = sorted({r["corruption"] for r in rows}); models = sorted({r["model"] for r in rows})
        clean = {r[0]: float(r[1]) for r in csv.reader(open(RESULTS / "robustness_summary.csv")) if r[0] != "model"}
        fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharey=True)
        for ax, kind in zip(axes.ravel(), kinds):
            for m in models:
                ys = [100 * clean[m]] + [100 * float(r["accuracy"]) for r in sorted((r for r in rows if r["model"] == m and r["corruption"] == kind), key=lambda r: int(r["severity"]))]
                ax.plot(range(len(ys)), ys, marker="o", ms=3, lw=1, label=m)
            ax.set_title(kind, fontsize=9); ax.set_xticks(range(4)); ax.set_xlabel("severity", fontsize=8); ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("accuracy (%)"); axes[0, 0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(FIGURES / "robustness_curves.pdf"); fig.savefig(FIGURES / "robustness_curves.png", dpi=200)

    # ---- markdown summary
    md = ["# MAPIMALA extended benchmark - summary", "", "## Main results (Table 3)", "",
          "| model | accuracy | 95% CI | P | R | F1 | F1 95% CI | seeds |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |" for r in main_rows]
    md += ["", "## Cost (Table 7)", "", "| model | params | train (min) | latency (ms) | size (MB) | acc (%) |", "|---|---|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in cost_rows]
    md += ["", f"Best model: **{best[0]}** ({100 * best[1]:.2f} %)", "", "## Ten most frequent confusions of the best model", ""]
    md += [f"- {c['true']} -> {c['predicted']}: {c['count']}" for c in cp]
    (RESULTS / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n-> results/summary_main.csv, cost.csv, mcnemar.csv, per_class.csv, confused_pairs.csv, summary.md; figures/pareto.pdf")


if __name__ == "__main__":
    main()
