# DE-ColonSegNet: Real-Time Polyp Detection in Clinical Endoscopy

This repository contains the implementation of **DE-ColonSegNet (Dual-Encoder ColonSegNet)**, an advanced semantic segmentation model tailored for real-time polyp detection in endoscopy. 

The architecture features a robust dual-encoder design intended to capture multi-scale spatial features effectively, ensuring both high recall for small lesions and fast inference speeds suitable for clinical settings.

## 🚀 Key Features
- **Dual-Encoder Architecture**: Enhances feature extraction and spatial context modeling.
- **Custom Loss Functions**: Combines a size-adaptive Tversky Loss (beta 0.85 for small lesions, 0.7 otherwise, to penalize false negatives) with Focal Loss (alpha 0.25, gamma 2.0). The auxiliary output of the first encoder is weighted by 0.4.
- **TensorRT Optimization**: Includes scripts to convert the PyTorch model to ONNX and TensorRT engines, accelerating inference on NVIDIA GPUs.
- **INT8 Quantization**: Post-training static quantization with ONNX Runtime (symmetric INT8 weights and activations, 50 calibration images). `fix_int8_onnx.py` restores only the convolution biases to FP32 by removing their DequantizeLinear nodes, which TensorRT failed to parse.

## 📁 Repository Structure
- `models/`: Model architecture (`de_colonsegnet.py`) and custom loss functions (`losses.py`).
- `datasets/`: Data loading pipelines and weighted sampling strategies (`coco_dataset.py`, `weighted_sampler.py`).
- `scripts/`: Data preparation and verification utilities.
- `train.py`: Main script for training the DE-ColonSegNet model.
- `evaluate.py`: Evaluation script to compute IoU, Dice coefficient, and Recall.
- `inference.py` / `inference_trt.py`: Inference scripts for standard PyTorch and optimized TensorRT models.
- `export_onnx.py` / `benchmark_trt.py`: Scripts for exporting to ONNX and benchmarking inference latency.
- `benchmark_trt_int8.py` / `fix_int8_onnx.py` / `run_int8_fixed.py`: INT8 PTQ calibration, bias QDQ fix, and INT8 engine benchmark.

## 🛠️ Environment Setup

We provide an `environment.yml` to easily set up the Conda environment.

```bash
conda env create -f environment.yml
conda activate de_colonsegnet_env
```

## 🏃‍♂️ How to Run

### 1. Training
To train the model from scratch, configure your dataset paths in `train.py` and run:
```bash
python train.py
```

### 2. Evaluation
To evaluate the best checkpoint on the test set:
```bash
python evaluate.py
```

### 3. Export & TensorRT Optimization
To export the trained PyTorch model to ONNX:
```bash
python export_onnx.py
```
To benchmark the model using TensorRT (FP16 or INT8):
```bash
python benchmark_trt.py
```

## ⚖️ License
This project is licensed under the MIT License.
