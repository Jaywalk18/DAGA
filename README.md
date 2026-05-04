# DAGA: Dynamic Attention-Guided Adaptation for Vision Foundation Models

<p align="center">
  <b>NeurIPS 2026 (Under Review)</b>
</p>

---

## Abstract

We propose **DAGA** (Dynamic Attention-Guided Adaptation), a parameter-efficient fine-tuning method for vision foundation models. DAGA leverages frozen backbone attention maps as spatial guidance signals to dynamically adapt features for downstream tasks. Our method achieves competitive performance across multiple vision tasks while introducing minimal trainable parameters.

---

## Method

<p align="center">
  <img src="assets/method.png" width="80%" alt="DAGA Architecture">
</p>

DAGA consists of three key components:

| Component | Description |
|-----------|-------------|
| **Attention Encoder** | Encodes multi-head attention maps into compact guidance vectors |
| **Dynamic Gate Generator** | Produces instance-specific gating signals conditioned on attention patterns |
| **Feature Transformer** | Applies gated transformations with residual connections |

---

## Installation

```bash
# Create environment
conda create -n dinov3_env python=3.11 -y
conda activate dinov3_env

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DINOv3
cd dinov3 && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download Pretrained Weights

```bash
mkdir -p checkpoints
# Download DINOv3 ViT-B/16 pretrained weights
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth -O checkpoints/dinov3_vitb16_pretrain.pth
```

### 2. Configure Paths

Edit `scripts/common_config.sh`:

```bash
CHECKPOINT_DIR="/path/to/checkpoints"
DATA_ROOT="/path/to/datasets"
DEFAULT_GPU_IDS="0,1,2"
```

### 3. Run Experiments

```bash
# Classification
bash scripts/run_classification.sh

# Detection
bash scripts/run_detection.sh

# Segmentation
bash scripts/run_segmentation.sh

# Depth Estimation
bash scripts/run_depth.sh
```

---

## Experiments

### Supported Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Classification | ImageNet-1K, CIFAR-100 | Top-1 Acc |
| Detection | COCO 2017 | mAP |
| Segmentation | ADE20K | mIoU |
| Depth Estimation | NYU Depth v2 | AbsRel, δ<1.25 |
| Robustness | ImageNet-C/A/R | mCE, Top-1 Acc |

### DAGA Layer Configuration

```bash
# Hourglass configuration (recommended)
--use_daga --daga_layers 1 2 10 11

# Deep layers only
--use_daga --daga_layers 8 9 10 11

# Full adaptation
--use_daga --daga_layers 0 1 2 3 4 5 6 7 8 9 10 11
```

### Hyperparameter Search

```bash
bash scripts/run_hyperparameter_search.sh
```

---

## Results

All numbers reported with DINOv3-ViT-B/16 backbone unless otherwise noted.

### Classification on ImageNet-1K

| Method | Params (M) | Top-1 Acc (%) |
|--------|------------|---------------|
| Baseline (Frozen) | 0.1 | 80.9 |
| Layer Fine-tuning | 22 | 85.9 |
| LoRA | 0.3 | 83.2 |
| AdaptFormer | 1.2 | 84.8 |
| ViT-Adapter | 1.5 | 85.1 |
| **DAGA (Ours)** | **1.2** | **85.9** |

Multi-seed (n=3): DAGA **85.77 ± 0.06**, the tightest seed-to-seed std among PEFT methods.

### Detection on COCO

| Method | Backbone | AP |
|--------|----------|-----|
| Baseline | DINOv3-B | 32.5 |
| AdaptFormer | DINOv3-B | 33.5 |
| ViT-Adapter | DINOv3-B | 32.8 |
| **DAGA (Ours)** | DINOv3-B | **35.6** |

### Segmentation on ADE20K

| Method | Backbone | mIoU |
|--------|----------|------|
| Baseline | DINOv3-B | 38.4 |
| AdaptFormer | DINOv3-B | 45.4 |
| ViT-Adapter | DINOv3-B | 47.0 |
| **DAGA (Ours)** | DINOv3-B | **50.5** |

### Depth Estimation on NYU Depth v2

| Method | RMSE ↓ | δ<1.25 ↑ |
|--------|----------|----------|
| Baseline | 0.469 | 88.0 |
| AdaptFormer | 0.446 | 90.4 |
| ViT-Adapter | 0.437 | 91.0 |
| **DAGA (Ours)** | **0.432** | **91.3** |

### Cross-Backbone Generalization

DAGA's gain tracks the semantic quality of the backbone's attention rather than pre-training scale:

| Backbone | Pre-training | IN-1K Δ | ADE20K mIoU Δ |
|---|---|---|---|
| DINOv3 | self-distillation | **+5.0** | **+12.1** |
| DINOv2 | self-distillation | **+5.0** | +10.9 |
| iBOT | self-distillation + MIM | +4.5 | +9.2 |
| MAE | masked image modelling | +0.2 | +0.4 |
| CLIP | image–text contrastive | +0.1 | +0.2 |
| DeiT | supervised w/ distillation | +0.1 | +0.2 |
| MoCo v3 | instance contrastive | −0.2 | −0.3 |

---

## Project Structure

```
Dino_DAGA/
├── core/                   # Core modules
│   ├── daga.py            # DAGA implementation
│   ├── backbones.py       # Backbone utilities
│   ├── heads.py           # Task heads
│   └── utils.py           # Utilities
├── tasks/                  # Task implementations
├── data/                   # Dataset loaders
├── scripts/                # Training scripts
│   ├── common_config.sh   # Shared configuration
│   ├── run_classification.sh
│   ├── run_detection.sh
│   ├── run_segmentation.sh
│   ├── run_depth.sh
│   └── run_hyperparameter_search.sh
├── visualization/          # Visualization tools
├── main_*.py              # Entry points
└── requirements.txt
```

---

## Acknowledgments

This work builds upon:
- [DINOv3](https://github.com/facebookresearch/dinov3) - Self-supervised Vision Transformer
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter) - Vision Transformer Adapter

---

## License

This project is released under the MIT License.
