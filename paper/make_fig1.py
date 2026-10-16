"""Figure 1: overview of the MAPIMALA benchmark (vector-quality PNG + PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = Path(__file__).resolve().parent.parent / "figures"

plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["pdf.fonttype"] = 42
fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=300)
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.0); ax.axis("off")

def box(x, y, w, h, title, lines, fc="#F4F6F8", ec="#2F4858", title_fc="#2F4858"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.52), w, 0.52, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=title_fc, ec=title_fc, lw=1.2))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")
    yy = y + h - 0.85
    for ln, style in lines:
        ax.text(x + 0.18, yy, ln, ha="left", va="center", fontsize=8.6,
                color="#1b1b1b", fontweight=("bold" if style == "b" else "normal"),
                fontstyle=("italic" if style == "i" else "normal"))
        yy -= 0.33

def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.6, color="#2F4858"))

# Stage 1 - data
box(0.2, 0.9, 2.9, 4.7, "1  Data", [
    ("DIMPSAR leaf images", "b"),
    ("5,945 images · 40 species", "n"),
    ("smartphone, field conditions", "i"),
    ("", "n"),
    ("Stratified split (seed 42)", "b"),
    ("train 4,161 / val 594 / test 1,190", "n"),
    ("", "n"),
    ("Pre-processing", "b"),
    ("resize 224² (conv.) / 64² (FC)", "n"),
    ("[0,1] scaling; augmentation", "n"),
    ("for fine-tuning only", "n"),
])
arrow(3.15, 3.55, 3.25)

# Stage 2 - representations / classifiers
box(3.6, 0.9, 3.6, 4.7, "2  Representations × classifiers", [
    ("(a) raw pixels → MLP (BP, DNN)", "n"),
    ("(b) PCA-100 → GA-initialised MLP", "n"),
    ("(c) CNN trained from scratch", "n"),
    ("(d) frozen ImageNet backbone", "n"),
    ("     + trained head (VGG16, …)", "n"),
    ("(e) fine-tuned backbones", "n"),
    ("     (MobileNetV3, EfficientNetV2,", "n"),
    ("     DenseNet121, ResNet50, ConvNeXt-T)", "n"),
    ("", "n"),
    ("Single CPU budget, no GPU", "b"),
])
arrow(7.25, 7.65, 3.25)

# Stage 3 - evaluation
box(7.7, 0.9, 3.5, 4.7, "3  Evaluation", [
    ("Accuracy, macro P/R/F1", "n"),
    ("Wilson & bootstrap 95 % CIs", "n"),
    ("McNemar pairwise tests", "n"),
    ("Per-class errors, confused pairs", "n"),
    ("", "n"),
    ("Cost: params, train time,", "b"),
    ("CPU latency, model size", "n"),
    ("Robustness: DIMPSAR-C", "b"),
    ("(acquisition corruptions)", "n"),
    ("Shortcut analysis: Grad-CAM,", "b"),
    ("leaf/background masking", "n"),
])
arrow(11.25, 11.65, 3.25)

# Stage 4 - deployment
box(11.7, 0.9, 1.6, 4.7, "4  Use", [
    ("Model", "b"),
    ("selection", "n"),
    ("", "n"),
    ("MAPIMALA", "b"),
    ("application", "n"),
    ("(capture /", "n"),
    ("upload,", "n"),
    ("species +", "n"),
    ("fact sheet)", "n"),
])

fig.savefig(OUT / "fig1_overview.png", bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(OUT / "fig1_overview.pdf", bbox_inches="tight", facecolor="white")
print("ok")
