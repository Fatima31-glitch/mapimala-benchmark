#!/usr/bin/env bash
# Reproduces every result of the paper. Usage: bash run_all.sh "/path/to/Medicinal plant dataset"
# Steps 3-4 are meant for a GPU (Colab T4); the others run on a CPU.
set -e
DATA="$1"; [ -z "$DATA" ] && { echo "usage: bash run_all.sh DATA_DIR"; exit 1; }
BB=vgg16,mobilenetv3l,efficientnetv2b0,densenet121,resnet50
python 01_extract_features.py --data "$DATA" --backbones $BB
python 02_probe_and_ablation.py --data "$DATA" --backbones $BB
python 03_finetune.py --data "$DATA" --backbones mobilenetv3l,efficientnetv2b0,densenet121 --seeds 42,43,44
python 03_finetune.py --data "$DATA" --backbones mobilenetv3l,efficientnetv2b0 --seeds 43 --no-augment
python 04_robustness.py --data "$DATA"
python 05_shortcut.py --data "$DATA"
python 06_make_tables.py --data "$DATA"
python 07_latency_cpu.py --data "$DATA"
python 08_bp_pixels.py --data "$DATA"
