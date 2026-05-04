# DAGA: Dynamic Attention-Guided Adaptation for Self-Supervised Vision Transformers

<p align="center">
  <b>NeurIPS 2026 (Under Review)</b>
</p>

<p align="center">
  Anonymous mirror for reviewers: <a href="https://anonymous.4open.science/r/DAGA-NeurIPS2026">anonymous.4open.science/r/DAGA-NeurIPS2026</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a>
</p>

---

## Abstract

Self-supervised Vision Transformers in the DINO family encode rich semantic structure within their self-attention maps, yet existing parameter-efficient fine-tuning (PEFT) methods apply *static, content-agnostic* transformations that ignore this internal signal. We present **DAGA** (*Dynamic Attention-Guided Adaptation*), a PEFT module that repurposes a frozen backbone's emergent attention maps as instance-specific guidance: per-image attention is converted into channel and spatial gates that condition a lightweight bottleneck adapter. With only **1.4% additional parameters**, DAGA reaches **85.9% top-1 on ImageNet-1K** with DINOv3-ViT-B (multi-seed mean **85.77 ± 0.06** over 3 seeds), surpassing recent PEFT baselines, with strong gains on dense prediction (**+12.1% mIoU** on ADE20K) and fine-grained classification (**+1.6 / +3.5** over ViT-Adapter on CUB-200 / FGVC-Aircraft). DAGA's effectiveness is tied to the semantic quality of the backbone's attention — large gains on DINOv2/v3 and iBOT, small on MAE / CLIP / DeiT — positioning it as the first PEFT module to operationalize emergent SSL attention as in-loop guidance, complementing self-distillation methods that refine attention itself.

---

## Method

DAGA consists of three tightly coupled components inserted between selected frozen backbone blocks:

| Component | Description | Params (DINOv3-ViT-B) |
|---|---|---|
| **Attention Encoder** | Shared 2-layer CNN over the frozen attention map; computed once per forward pass and cached across DAGA layers | 19 K |
| **Layer-Specific Heads** | Per-DAGA-layer MLP that produces a 128-d guidance vector + a 1×1 spatial weight from the cached encoder output | 4 × 25 K |
| **Adaptation Layer** | Channel gate (Linear 128→D + Sigmoid) and bottleneck MLP (D→107→D), combined as `x + α · (G_c · (1 + β · S)) · Ψ(x)` with learnable α, β | 4 × 264 K |
| **Total** | | **1.20 M** (1.4% of 86 M backbone) |

The full architecture and pseudocode are in Appendix F–G of the paper.

---

## Installation

```bash
# Create environment
conda create -n dinov3_env python=3.11 -y
conda activate dinov3_env

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DINOv3 (clone facebookresearch/dinov3 into ./dinov3 first)
cd dinov3 && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download Pretrained Weights

```bash
mkdir -p checkpoints
# DINOv3 ViT-B/16 pretrained on LVD-1689M
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    -O checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
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
# Image classification (matched-protocol benchmarks, paper Table 1)
bash scripts/run_classification.sh

# Object detection on COCO (paper Table 3)
bash scripts/run_detection.sh

# Semantic segmentation on ADE20K (paper Table 3)
bash scripts/run_segmentation.sh

# Depth estimation on NYU Depth v2 (paper Table 3)
bash scripts/run_depth.sh

# Multi-seed reproducibility (paper Table 6)
bash scripts/run_multiseed.sh
```

See `EXPERIMENT_GUIDE.md` for detailed dataset paths, Optuna sweep results, and per-task hyperparameters used in the paper.

---

## Experiments

### Supported Tasks

| Task | Datasets | Metric |
|------|---------|--------|
| Classification (standard) | ImageNet-1K, CIFAR-10/100, Flowers-102, Oxford Pets, Food-101, Stanford Cars, DTD, SUN397 | Top-1 Acc |
| Classification (fine-grained) | CUB-200, FGVC-Aircraft | Top-1 Acc |
| Detection | COCO 2017 | AP, AP₅₀, AP₇₅, APₛ |
| Segmentation | ADE20K, VOC2012, Cityscapes | mIoU, aAcc |
| Depth Estimation | NYU Depth v2 | RMSE, REL, δ₁, δ₂ |
| Retrieval | ROxford5k, RParis6k | mAP (Easy/Med/Hard) |
| Robustness | ImageNet-C | Top-1 Acc per corruption |

### DAGA Layer Configuration

Optuna sweep results (per the paper, Section 4 and Appendix I):

| Task | Best LR | Best DAGA layers |
|---|---|---|
| Classification & Retrieval | task-dependent (1e-4 to 5e-3) | hourglass `{1, 2, 10, 11}` |
| Detection | 1e-3 | hourglass `{1, 2, 10, 11}` |
| Segmentation | 5e-3 | deep `{8, 9, 10, 11}` |
| Depth | 3.7e-4 | spread `{2, 5, 8, 11}` |

```bash
# Hourglass placement (recommended for classification / detection / retrieval)
--use_daga --daga_layers 1 2 10 11

# Deep placement (recommended for dense prediction)
--use_daga --daga_layers 8 9 10 11
```

---

## Results

All numbers reported with DINOv3-ViT-B/16 backbone unless otherwise noted; full per-method / per-dataset breakdowns are in the paper.

### Image Classification (paper Table 1)

| Method | Params | IN-1K | CIFAR-100 | Flowers | Pets | Cars | DTD | CUB-200 | FGVC | Avg (9-std) |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear Probe | 0.1 M | 80.9 | 84.3 | 99.3 | 95.7 | 88.5 | 76.2 | — | — | 86.5 |
| Full Fine-tune | 85.7 M | 84.4 | 92.1 | 99.4 | 95.6 | 90.5 | 77.8 | — | — | 90.4 |
| Layer-FT | 22 M | 85.9 | 93.5 | **99.7** | 96.4 | 92.3 | 79.5 | — | — | 91.7 |
| LoRA | 0.3 M | 83.2 | 90.5 | 98.2 | 94.1 | 88.5 | 75.8 | 86.77 | 77.50 | 88.9 |
| VPT-Deep | 0.6 M | 83.1 | 89.1 | 85.1 | 92.8 | 85.2 | 73.8 | — | — | 85.9 |
| AdaptFormer | 1.2 M | 84.8 | 91.8 | 98.8 | 95.2 | 91.2 | 78.5 | 87.58 | 78.76 | 90.5 |
| ViT-Adapter | 1.5 M | 85.1 | 92.5 | 99.2 | 95.8 | 92.5 | 79.8 | 87.89 | 82.30 | 91.1 |
| **DAGA (Ours)** | **1.2 M** | **85.9** | **93.6** | 99.6 | **96.5** | **94.9** | **82.3** | **89.46** | **85.75** | **92.2** |

**Multi-seed ImageNet-1K** (n=3 random seeds, paper Table 6): DAGA reaches **85.77 ± 0.06**, with the tightest seed-to-seed std among all PEFT methods (LoRA ±0.08, AdaptFormer ±0.13, Layer-FT ±0.07).

### Dense Prediction (paper Table 3)

| Method | Params | COCO AP | ADE20K mIoU | NYU RMSE↓ | NYU δ₁↑ | IN-C mean |
|---|---|---|---|---|---|---|
| Linear Probe | 0.5 M | 32.5 | 38.4 | 0.469 | 88.0 | 64.6 |
| Full-FT | 85.7 M | 33.1 | 48.1 | 0.548 | 84.8 | 61.1 |
| Layer-FT | 22 M | 35.4 | 50.3 | **0.414** | **92.5** | 64.2 |
| LoRA | 0.8 M | 32.2 | 44.1 | 0.457 | 89.6 | 61.2 |
| AdaptFormer | 1.2 M | 33.5 | 45.4 | 0.446 | 90.4 | 63.1 |
| ViT-Adapter | 1.5 M | 32.8 | 47.0 | 0.437 | 91.0 | 63.8 |
| **DAGA (Ours)** | **1.2 M** | **35.6** | **50.5** | 0.432 | 91.3 | **65.3** |

### Cross-Backbone Generalization (paper Table 4)

DAGA's effectiveness tracks the semantic quality of the backbone's CLS-to-patch attention rather than the size of the pre-training corpus:

| Backbone | Pre-training | #Imgs | IN-1K Base → +DAGA (Δ) | ADE20K Base → +DAGA (Δ) |
|---|---|---|---|---|
| DINOv3 | self-distillation | 1689 M | 80.9 → 85.9 (**+5.0**) | 38.4 → 50.5 (**+12.1**) |
| DINOv2 | self-distillation | 142 M | 79.6 → 84.6 (**+5.0**) | 36.9 → 47.8 (+10.9) |
| iBOT | self-distillation + MIM | 14 M | 79.4 → 83.9 (+4.5) | 35.5 → 44.7 (+9.2) |
| MAE | masked image modelling | 1.3 M | 68.0 → 68.2 (+0.2) | 30.1 → 30.5 (+0.4) |
| CLIP | image–text contrastive | 400 M | 76.5 → 76.6 (+0.1) | 32.6 → 32.8 (+0.2) |
| DeiT | supervised w/ distillation | 1.3 M | 81.8 → 81.9 (+0.1) | 34.2 → 34.4 (+0.2) |
| MoCo v3 | instance contrastive | 1.3 M | 76.7 → 76.5 (−0.2) | 31.8 → 31.5 (−0.3) |

DAGA contributes large gains only on backbones whose CLS-to-patch attention is already object-centric and artifact-free (the DINO/iBOT family); on backbones with diffuse or noisy attention (MAE, CLIP, DeiT, MoCo v3) the contribution is marginal. This makes the dependency on attention quality explicit rather than implicit.

---

## Project Structure

```
Dino_DAGA/
├── core/                       # Core modules
│   ├── daga.py                 # DAGA implementation (Attention Encoder + Heads + Adapter)
│   ├── backbones.py            # DINOv3/v2/iBOT/MAE/CLIP/DeiT/MoCo wrappers
│   ├── heads.py                # Task heads (cls / det / seg / depth / retrieval)
│   └── utils.py                # Utilities
├── data/                       # Dataset loaders
├── dinov3/                     # DINOv3 source (must be cloned separately)
├── scripts/                    # Training entry-point shell scripts
│   ├── common_config.sh        # Shared paths and GPUs
│   ├── run_classification.sh
│   ├── run_detection.sh
│   ├── run_segmentation.sh
│   ├── run_depth.sh
│   ├── run_multiseed.sh
│   └── run_hyperparameter_search.sh  # Optuna sweep
├── paper_experiments/          # 8-PEFT-baseline comparison harness
├── experiments/                # Per-task per-dataset markdown reports
├── visualization/              # Attention vis + failure-case scripts
├── main_*.py                   # Per-task entry points
├── EXPERIMENT_GUIDE.md         # Full dataset paths + Optuna sweep results
├── CHECKPOINT_STATUS.md        # Status of trained checkpoints
└── requirements.txt
```

---

## Citation

```bibtex
@inproceedings{anonymous2026daga,
  title={DAGA: Dynamic Attention-Guided Adaptation for Self-Supervised Vision Transformers},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Under review}
}
```

---

## Acknowledgments

This work builds upon:
- [DINOv3](https://github.com/facebookresearch/dinov3) — Self-supervised Vision Transformer
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter) — Vision Transformer Adapter
- [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), [LoRA](https://github.com/microsoft/LoRA), [VPT](https://github.com/KMnP/vpt) — PEFT baselines used in the matched-protocol comparison

---

## License

This project is released under the MIT License.
