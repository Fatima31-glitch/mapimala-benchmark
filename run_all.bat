@echo off
REM Reproduces every result of the paper. Usage: run_all.bat "C:\path\to\Medicinal plant dataset"
set DATA=%~1
set BB=vgg16,mobilenetv3l,efficientnetv2b0,densenet121,resnet50
python 01_extract_features.py --data "%DATA%" --backbones %BB% || exit /b 1
python 02_probe_and_ablation.py --data "%DATA%" --backbones %BB% || exit /b 1
python 03_finetune.py --data "%DATA%" --backbones mobilenetv3l,efficientnetv2b0,densenet121 --seeds 42,43,44 || exit /b 1
python 03_finetune.py --data "%DATA%" --backbones mobilenetv3l,efficientnetv2b0 --seeds 43 --no-augment || exit /b 1
python 04_robustness.py --data "%DATA%" || exit /b 1
python 05_shortcut.py --data "%DATA%" || exit /b 1
python 06_make_tables.py --data "%DATA%" || exit /b 1
python 07_latency_cpu.py --data "%DATA%" || exit /b 1
python 08_bp_pixels.py --data "%DATA%" || exit /b 1
