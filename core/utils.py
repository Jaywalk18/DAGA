import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
from datetime import date
import os

# Import swanlab only if not disabled (for visualization scripts that don't need it)
swanlab = None
if os.environ.get('SWANLAB_DISABLED') != '1':
    try:
        import swanlab as _swanlab
        swanlab = _swanlab
    except Exception as e:
        print(f"Warning: Could not import swanlab: {e}")


def setup_environment(seed):
    """Setup random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_model(model):
    """Get the base model, handling DataParallel/DDP wrapper"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(model, (DataParallel, DDP)):
        return model.module
    return model


def save_checkpoint(model, optimizer, epoch, best_metric, args, path, vis_data=None):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": get_base_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
    }
    if vis_data is not None:
        checkpoint["visualization_images"] = vis_data.cpu()
    torch.save(checkpoint, path)


def setup_logging(args, task_name="classification"):
    """Setup SwanLab logging"""
    import os
    
    # Support both use_daga (original) and method (comparison) styles
    if hasattr(args, 'method'):
        # Paper comparison style
        method_name = args.method
        layers_str = '-'.join(map(str, args.adaptation_layers))
    else:
        # Original style
        method_name = "daga" if args.use_daga else "baseline"
        layers_str = '-'.join(map(str, args.daga_layers)) if args.use_daga else ''
    
    exp_name = (
        args.swanlab_name
        or f"{args.dataset}_{method_name}_L{layers_str}_{date.today()}"
    )
    
    # Check both environment variable and args attribute
    swanlab_mode = os.environ.get('SWANLAB_MODE', '').lower()
    enable_swanlab = getattr(args, 'enable_swanlab', False) and swanlab_mode != 'disabled'
    
    # Store enable_swanlab in args for later use
    args.enable_swanlab = enable_swanlab
    
    if enable_swanlab:
        # Define project names for different tasks
        # For classification, use dataset name in project
        dataset_name = getattr(args, 'dataset', 'unknown').upper()
        
        # Map dataset names for better project naming
        depth_dataset_map = {
            "nyu_depth_v2": "NYUv2",
            "kitti": "KITTI",
        }
        depth_project_name = depth_dataset_map.get(getattr(args, 'dataset', ''), dataset_name)
        
        seg_dataset_map = {
            "ade20k": "ADE20K",
            "voc2012": "VOC2012",
            "cityscapes": "Cityscapes",
        }
        seg_project_name = seg_dataset_map.get(getattr(args, 'dataset', ''), dataset_name)
        
        retrieval_dataset_map = {
            "roxford5k": "ROxford",
            "rparis6k": "RParis",
        }
        retrieval_project_name = retrieval_dataset_map.get(getattr(args, 'dataset', ''), dataset_name)
        
        project_mapping = {
            "classification": f"DINOv3-{dataset_name}-Classification",
            "dinotxt": "DINOv3-COCO-TextImageAlignment",
            "detection": "DINOv3-COCO-Detection", 
            "segmentation": f"DINOv3-{seg_project_name}-Segmentation",
            "depth": f"DINOv3-{depth_project_name}-Depth",
            "retrieval": f"DINOv3-{retrieval_project_name}-Retrieval",
            "robustness": "DINOv3-ImageNet-C-Robustness",
            "linear": "DINOv3-Linear-Probing",
            "logreg": "DINOv3-Logistic-Regression",
            "knn": "DINOv3-KNN-Evaluation",
            "paper_comparison": f"DINOv3-{dataset_name}-Classification",
            "detection_comparison": f"DINOv3-{dataset_name}-Detection",
        }
        
        project_name = project_mapping.get(task_name, f"DINOv3-{task_name}")
        
        if swanlab is not None:
            init_kwargs = {
                "workspace": "Dino_DAGA",
                "project": project_name,
                "experiment_name": exp_name,
                "config": vars(args),
            }
            if not swanlab_mode:
                init_kwargs["mode"] = getattr(args, 'swanlab_mode', 'disabled')
            swanlab.init(**init_kwargs)
        else:
            print("Warning: swanlab not available, logging disabled")
    return exp_name


def create_dataloaders(train_dataset, test_dataset, batch_size, num_workers=8):
    """Create train and test dataloaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to avoid DataParallel issues
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def finalize_experiment(best_metric, final_metric, total_time_minutes, output_dir, metric_name="Acc", enable_swanlab=True):
    """Print final results and close SwanLab"""
    print(f'\n{"="*70}\n🎉 Training Completed!\n{"="*70}')
    print(f"Total Time:       {total_time_minutes:.1f} minutes")
    print(f"Best {metric_name}:    {best_metric:.2f}%")
    print(f"Final {metric_name}:   {final_metric:.2f}%")
    print(f"Results saved to: {output_dir}\n{'='*70}\n")
    if enable_swanlab and swanlab is not None:
        swanlab.finish()


class TopKCheckpointManager:
    """Manage top-K checkpoints based on metric value"""
    
    def __init__(self, output_dir, k=3, metric_higher_is_better=True):
        self.output_dir = Path(output_dir)
        self.k = k
        self.higher_is_better = metric_higher_is_better
        self.checkpoints = []  # List of (metric, epoch, path)
    
    def update(self, model, optimizer, epoch, metric, args):
        """Update top-K checkpoints"""
        # Save current checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch+1}_metric{metric:.4f}.pth"
        save_checkpoint(model, optimizer, epoch, metric, args, ckpt_path)
        self.checkpoints.append((metric, epoch, ckpt_path))
        
        # Sort by metric
        self.checkpoints.sort(key=lambda x: x[0], reverse=self.higher_is_better)
        
        # Keep only top-K
        while len(self.checkpoints) > self.k:
            _, _, path_to_remove = self.checkpoints.pop()
            if path_to_remove.exists():
                path_to_remove.unlink()
        
        # Update best_model.pth symlink or copy
        if self.checkpoints:
            best_metric, _, best_path = self.checkpoints[0]
            best_link = self.output_dir / "best_model.pth"
            if best_link.exists():
                best_link.unlink()
            # Copy instead of symlink for portability
            import shutil
            shutil.copy(best_path, best_link)
            return True, best_metric
        return False, None
    
    def save_final(self, model, optimizer, epoch, metric, args):
        """Save final epoch checkpoint"""
        final_path = self.output_dir / "final_model.pth"
        save_checkpoint(model, optimizer, epoch, metric, args, final_path)


def save_results_json(output_dir, results_dict):
    """Save experiment results to JSON file"""
    import json
    output_path = Path(output_dir) / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    return output_path


def load_results_json(output_dir):
    """Load experiment results from JSON file"""
    import json
    results_path = Path(output_dir) / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None
