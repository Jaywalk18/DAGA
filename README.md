# DAGA: Dynamic Attention-Guided Adapter for Self-Supervised Vision Transformers

<p align="center">
  <b>ICML 2026 Submission</b>
</p>

<p align="center">
  <img src="assets/architecture.png" width="800" alt="DAGA Architecture">
</p>

## Overview

**DAGA** (Dynamic Attention-Guided Adapter) is a parameter-efficient fine-tuning method that repurposes self-attention maps from frozen Vision Transformers as instance-specific guidance signals for feature adaptation.

### Key Features

- **Attention-as-Guidance**: Leverages frozen backbone attention maps as dynamic, content-aware signals
- **Instance-Specific Adaptation**: Each input receives unique gating based on its semantic layout  
- **Learnable Spatial Scale**: Automatically discovers task-appropriate spatial weighting
- **Parameter Efficient**: Only 1.4% additional parameters over frozen backbone

## Installation

```bash
# Create environment
conda create -n daga python=3.11 -y
conda activate daga

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install DINOv3 (as local package)
cd dinov3 && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Pretrained Weights

```bash
mkdir -p checkpoints
# Download DINOv3 ViT-B/14 pretrained weights from official source
```

### 2. Configure Paths

Edit `scripts/common_config.sh`:

```bash
CHECKPOINT_DIR="/path/to/checkpoints"
DATA_ROOT="/path/to/datasets"
GPU_IDS="0,1"
```

### 3. Run Experiments

```bash
# Classification (fine-tuning)
bash scripts/run_classification.sh

# Object Detection (COCO)
bash scripts/run_detection.sh

# Semantic Segmentation (ADE20K)
bash scripts/run_segmentation.sh

# Depth Estimation (NYU Depth V2)
bash scripts/run_depth.sh

# Feature Evaluation (frozen features)
bash scripts/run_knn.sh          # KNN classification
bash scripts/run_linear.sh       # Linear probe
bash scripts/run_logreg.sh       # Logistic regression

# Instance Retrieval (ROxford, RParis)
bash scripts/run_retrieval.sh

# Robustness Evaluation (ImageNet-C)
bash scripts/run_robustness.sh

# Vision-Language (DINOtxt)
bash scripts/run_dinotxt.sh
```

## Usage

### Basic DAGA Integration

```python
from core.daga import DAGA, create_daga

# Create DAGA module
daga = create_daga(
    feature_dim=768,           # ViT embedding dimension
    daga_layers=[1, 2, 10, 11], # Layers to apply DAGA (hourglass config)
    mlp_ratio=0.25,
    drop_rate=0.1,
)

# During forward pass
# attention_map: (B, H, W) from frozen ViT
# patch_features: (B, N, D) patch token features
adapted_features = daga.apply(patch_features, attention_map, layer_idx, H, W)
```

### DAGA Layer Configurations

```bash
# Hourglass (recommended) - shallow + deep layers
--use_daga --daga_layers 1 2 10 11

# Deep layers only
--use_daga --daga_layers 8 9 10 11

# Full adaptation (all layers)
--use_daga --daga_layers 0 1 2 3 4 5 6 7 8 9 10 11
```

## Project Structure

```
DAGA/
├── core/
│   ├── daga.py              # DAGA implementation
│   ├── backbones.py         # DINOv3 backbone utilities
│   ├── heads.py             # Task-specific heads
│   ├── datasets/            # Dataset loaders
│   └── utils.py             # Common utilities
├── data/                    # Dataset definitions
├── tasks/                   # Task implementations
├── scripts/                 # Training & evaluation scripts
├── main_classification.py   # Fine-tuning classification
├── main_detection.py        # Object detection
├── main_segmentation.py     # Semantic segmentation
├── main_depth.py            # Depth estimation
├── main_knn.py              # KNN evaluation
├── main_linear.py           # Linear probe
├── main_logreg.py           # Logistic regression
├── main_retrieval.py        # Instance retrieval
├── main_robustness.py       # Robustness (ImageNet-C)
├── main_dinotxt.py          # Vision-language alignment
└── requirements.txt
```

## Supported Tasks & Datasets

| Task | Datasets | Metrics |
|------|----------|---------|
| Classification | CIFAR-10/100, ImageNet-1K, Food-101, Flowers-102, Pets, Cars, SUN397, DTD | Top-1 Accuracy |
| KNN / Linear / LogReg | ImageNet-1K | Top-1 Accuracy |
| Detection | COCO 2017 | mAP |
| Segmentation | ADE20K, VOC2012, Cityscapes | mIoU |
| Depth Estimation | NYU Depth V2 | AbsRel, δ<1.25 |
| Instance Retrieval | ROxford5k, RParis6k | mAP |
| Robustness | ImageNet-C | mCE |
| Vision-Language | COCO Captions | Retrieval R@K |

## Visualization

DAGA produces more semantically-focused attention compared to baseline frozen features:

<p align="center">
  <img src="assets/visualization.png" width="800" alt="Feature Visualization">
</p>

## License

This project is released under the MIT License.

