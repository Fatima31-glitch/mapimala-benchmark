"""Figures 5 (accuracy vs CPU latency, Pareto front) and 6 (DIMPSAR-C curves) of the paper, from ../results. Output: ../figures/fig5_cost.*, ../figures/fig6_robustness.*"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE.parent / "results"
OUT = HERE.parent / "figures"
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "ps.fonttype": 42, "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#555555",
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5})
# validated categorical palette (slots 1-6) + neutral grey for the scratch CNN
COL = {"vgg16": "#2a78d6", "mobilenetv3l": "#eb6834", "efficientnetv2b0": "#1baf7a", "densenet121": "#6250d6", "resnet50": "#eda100", "cnn": "#777777"}
NAME = {"vgg16": "VGG16", "mobilenetv3l": "MobileNetV3-L", "efficientnetv2b0": "EfficientNetV2-B0", "densenet121": "DenseNet121", "resnet50": "ResNet50"}
MARK = {"vgg16": "o", "mobilenetv3l": "s", "efficientnetv2b0": "^", "densenet121": "D", "resnet50": "v"}

evals = {p.stem[5:]: json.load(open(p)) for p in RES.glob("eval_*.json")}

# ------------------------------------------------------------ Fig 5: accuracy vs CPU latency (Pareto)
lat = pd.read_csv(RES / "latency_cpu.csv")
LAT = {r.model: float(r.latency_ms_median) for r in lat.itertuples() if r.model != "hardware"}
fig, ax = plt.subplots(figsize=(4.7, 3.3), dpi=300)
ax.grid(True, axis="y"); ax.set_axisbelow(True)
pts = []
e = evals["cnn_scratch"]
ax.scatter(LAT["cnn_scratch"], 100 * e["accuracy"], marker="x", s=36, color=COL["cnn"], linewidths=1.2, zorder=3)
ax.annotate("CNN from scratch", (LAT["cnn_scratch"], 100 * e["accuracy"]), xytext=(7, -3), textcoords="offset points", fontsize=7, color="#444444")
pts.append((LAT["cnn_scratch"], 100 * e["accuracy"]))
for bb in ["vgg16", "mobilenetv3l", "efficientnetv2b0", "densenet121", "resnet50"]:
    f = evals[f"{bb}_frozen"]; xf = LAT[f"{bb}_frozen"]
    ax.scatter(xf, 100 * f["accuracy"], marker=MARK[bb], s=34, facecolors="white", edgecolors=COL[bb], linewidths=1.2, zorder=3)
    pts.append((xf, 100 * f["accuracy"]))
    seeds = [f"{bb}_ft_seed{s}" for s in (42, 43, 44) if f"{bb}_ft_seed{s}" in evals]
    if seeds:
        accs = np.array([evals[n]["accuracy"] for n in seeds]) * 100
        xt = np.mean([LAT[n] for n in seeds])
        ax.errorbar(xt, accs.mean(), yerr=accs.std(ddof=1), fmt=MARK[bb], ms=6, color=COL[bb], ecolor=COL[bb], elinewidth=0.9, capsize=2.5, zorder=4)
        ax.plot([xf, xt], [100 * f["accuracy"], accs.mean()], color=COL[bb], lw=0.8, ls=":", zorder=2)
        pts.append((xt, accs.mean()))
# Pareto front (lower latency and higher accuracy)
pts.sort(); front = []; best = -1
for x, y in pts:
    if y > best: front.append((x, y)); best = y
lab = {"vgg16": (0, -13, "center", "frozen"), "mobilenetv3l": (8, 2, "left", "ft"), "efficientnetv2b0": (8, -2, "left", "ft"),
       "densenet121": (0, 9, "center", "ft"), "resnet50": (7, -9, "left", "frozen")}
for bb, (dx, dy, ha, which) in lab.items():
    if which == "ft":
        seeds = [f"{bb}_ft_seed{s}" for s in (42, 43, 44)]
        xy = (np.mean([LAT[n] for n in seeds]), 100 * np.mean([evals[n]["accuracy"] for n in seeds]))
    else:
        xy = (LAT[f"{bb}_frozen"], 100 * evals[f"{bb}_frozen"]["accuracy"])
    ax.annotate(NAME[bb], xy, xytext=(dx, dy), textcoords="offset points", fontsize=7, color="#333333", ha=ha)
ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([100, 150, 200, 300, 500, 700]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(90, 800)
ax.set_xlabel("CPU inference latency (ms per image, batch 1, median)")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(60, 100)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", color="#333333", markerfacecolor="white", ls="", ms=6, label="frozen backbone + head"),
     Line2D([], [], marker="o", color="#333333", ls="", ms=6, label="fine-tuned (mean ± sd, 3 seeds)"),
     Line2D([], [], marker="x", color=COL["cnn"], ls="", ms=6, label="trained from scratch")]
ax.legend(handles=h, loc="center right", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig5_cost.png"); fig.savefig(OUT / "fig5_cost.pdf")
print("pareto front:", front)

# ------------------------------------------------------------ Fig 6: DIMPSAR-C curves
rb = pd.read_csv(RES / "robustness.csv")
rs = pd.read_csv(RES / "robustness_summary.csv").set_index("model")
STALE = set()
corrs = ["gaussian_blur", "motion_blur", "gaussian_noise", "jpeg", "brightness", "contrast", "rotation", "zoom", "occlusion"]
TITLE = {"gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur", "gaussian_noise": "Gaussian noise", "jpeg": "JPEG compression",
         "brightness": "Brightness", "contrast": "Contrast", "rotation": "Rotation", "zoom": "Zoom / crop", "occlusion": "Occlusion"}


def curve(models, corr):
    sub = rb[(rb.model.isin(models)) & (rb.corruption == corr)].groupby("severity").accuracy.mean()
    clean = rs.loc[models].clean_accuracy.mean()
    return [0, 1, 2, 3], [100 * clean] + [100 * sub[s] for s in (1, 2, 3)]


series = [("cnn", ["cnn_scratch"], "CNN from scratch", ":", "x"),
          ("vgg16", ["vgg16_frozen"], "VGG16 frozen", "--", "o"),
          ("mobilenetv3l", ["mobilenetv3l_frozen"], "MobileNetV3-L frozen", "--", "s"),
          ("mobilenetv3l", [f"mobilenetv3l_ft_seed{s}" for s in (42, 43, 44)], "MobileNetV3-L fine-tuned", "-", "s"),
          ("efficientnetv2b0", ["efficientnetv2b0_frozen"], "EfficientNetV2-B0 frozen", "--", "^"),
          ("efficientnetv2b0", [f"efficientnetv2b0_ft_seed{s}" for s in (42, 43, 44)], "EfficientNetV2-B0 fine-tuned", "-", "^"),
          ("densenet121", ["densenet121_frozen"], "DenseNet121 frozen", "--", "D"),
          ("densenet121", [f"densenet121_ft_seed{s}" for s in (42, 43, 44)], "DenseNet121 fine-tuned", "-", "D")]
fig, axes = plt.subplots(3, 3, figsize=(6.85, 5.6), dpi=300, sharex=True, sharey=True)
for ax, corr in zip(axes.ravel(), corrs):
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    for key, models, labl, ls, mk in series:
        x, y = curve(models, corr)
        ax.plot(x, y, ls=ls, marker=mk, ms=3, lw=1.1, color=COL[key], label=labl, markerfacecolor="white" if ls == "--" else COL[key], markeredgewidth=0.9)
    ax.set_title(TITLE[corr], fontsize=8, pad=4)
    ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(0, 100)
for ax in axes[-1]: ax.set_xlabel("Severity (0 = clean)")
for ax in axes[:, 0]: ax.set_ylabel("Accuracy (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig6_robustness.png"); fig.savefig(OUT / "fig6_robustness.pdf")
print("figures written")
