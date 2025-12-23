# Hybrid Optimization Experiment (Structural Pruning + INT8 QAT) — CIFAR-10

This repo reproduces a simple end-to-end experiment:
1) Train a baseline model on CIFAR-10
2) Apply **structural pruning** (channel pruning) and fine-tune
3) Apply **INT8 QAT** (PyTorch eager mode) and fine-tune
4) Evaluate accuracy + benchmark latency + measure model size

> Notes:
- INT8 quantization here targets **CPU inference** (fbgemm). This is normal for a reproducible academic experiment.
- If you want **TensorRT / TFLite** deployment afterwards, export to ONNX/TFLite after steps 2–3.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 1) Baseline training
```bash
python train.py --arch resnet18 --epochs 25 --out checkpoints/baseline.pt
python evaluate.py --ckpt checkpoints/baseline.pt
python benchmark.py --ckpt checkpoints/baseline.pt
```

### 2) Structural pruning + fine-tune
```bash
python prune.py --in_ckpt checkpoints/baseline.pt --out checkpoints/pruned.pt --prune_ratio 0.6
python train.py --resume checkpoints/pruned.pt --epochs 10 --out checkpoints/pruned_ft.pt
python evaluate.py --ckpt checkpoints/pruned_ft.pt
python benchmark.py --ckpt checkpoints/pruned_ft.pt
```

### 3) QAT INT8 + fine-tune + convert
```bash
python qat.py --in_ckpt checkpoints/pruned_ft.pt --out checkpoints/pruned_qat_int8.pt --epochs 8
python evaluate.py --ckpt checkpoints/pruned_qat_int8.pt --int8
python benchmark.py --ckpt checkpoints/pruned_qat_int8.pt --int8
```

## Output
- `results.csv` is appended after each run if you pass `--log results.csv` to evaluate/benchmark.
