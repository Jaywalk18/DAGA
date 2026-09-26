# DAGA

**Dynamic Attention-Guided Adaptation for Self-Supervised Vision Transformers**

Tianjian Zhou, Jie Jiang, Yishan Li, Yifei Zhang · NeurIPS 2026 Main Track (poster)

DAGA uses attention from a frozen vision transformer to guide a small, input-dependent adapter. This repository contains the PyTorch implementation, task entry points, and comparison runners.

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="DAGA architecture">
</p>

## Setup

1. Create a Python 3.10 environment and install PyTorch for your hardware.
2. Clone this repository, then install [DINOv3](https://github.com/facebookresearch/dinov3) in `./dinov3`:

```bash
git clone https://github.com/Jaywalk18/DAGA.git
cd DAGA
git clone https://github.com/facebookresearch/dinov3.git
pip install -e ./dinov3
pip install -r requirements.txt
```

3. Download the official DINOv3 ViT-B/16 weights following the DINOv3 instructions. Put the weight file in `./checkpoints/`, or set `CHECKPOINT_DIR` to its directory. The default filename is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`; override it with `PRETRAINED_PATH` if needed.

Datasets and pretrained weights are not included. Each launch script reads its dataset location from an environment variable, so no source edits or machine-specific paths are required.

## Run

For a single-task example, point `DATA_PATH` at the dataset root:

```bash
DATA_PATH=/datasets/cifar100 bash scripts/run_classification.sh
```

Other task scripts are under `scripts/`. The comparison runners used for the paper are under `paper_experiments/`; for example:

```bash
DAGA_IMAGENET_PATH=/datasets/imagenet bash paper_experiments/scripts/run_comparison_classification.sh
```

The comparison scripts read `DAGA_IMAGENET_PATH`, `DAGA_COCO_PATH`, `DAGA_ADE20K_PATH`, `DAGA_NYU_PATH`, `DAGA_SUN397_PATH`, or `DAGA_IMAGENET_C_PATH` as applicable. `GPU_IDS`, `CHECKPOINT_DIR`, and `PRETRAINED_PATH` can be set in the environment. The shell runners are starting points for the listed task protocols; inspect their dataset layout, batch size, and training settings before use.

Cloud experiment tracking is off by default. Set `SWANLAB_MODE=cloud` (and `ENABLE_SWANLAB=1` for the classification comparison shell runner) only if you want to send a run to SwanLab.

Use the corresponding `paper_experiments/main_comparison_*.py` entry point when comparing with a paper table. The standalone `main_*.py` programs are task examples and can have different defaults. This release does not include downstream checkpoints or the original training logs.

The attention-encoding cache in this release includes a correctness fix for successive images with the same shape. The full benchmark suite has not been rerun after that fix.

## Code layout

- `core/daga.py`: attention encoder, gates, and adapter
- `core/backbones.py`: DINOv3 loading and attention extraction
- `tasks/` and `main_*.py`: task implementations and entry points
- `paper_experiments/`: paired DAGA and baseline runners
- `scripts/`: single-task launch scripts
- `visualization/`: figure and attention-map utilities

Run the CPU regression checks with `python -m unittest discover -s tests -v`.

## Citation

```bibtex
@inproceedings{zhou2026daga,
  title     = {DAGA: Dynamic Attention-Guided Adaptation for Self-Supervised Vision Transformers},
  author    = {Zhou, Tianjian and Jiang, Jie and Li, Yishan and Zhang, Yifei},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff). The paper will be linked here when the proceedings version is available.

## License

[MIT](LICENSE)
