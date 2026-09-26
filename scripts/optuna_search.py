#!/usr/bin/env python3
"""
Optuna-based hyperparameter search for DAGA
Directly calls main_*.py scripts with dynamic parameters
"""
import os
import sys
import argparse
import subprocess
import optuna
from optuna.trial import TrialState
from pathlib import Path
import json
import re
from tqdm import tqdm

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

PROJECT_ROOT = Path(__file__).parent.parent

# Default batch sizes (can be overridden via environment variables)
DEFAULT_BATCH_SIZES = {
    "classification": 1024,
    "detection": 128,
    "segmentation": 48,
    "depth": 12,
}

def get_batch_size(task):
    env_var = f"BS_{task.upper()}"
    return int(os.environ.get(env_var, DEFAULT_BATCH_SIZES[task]))

TASK_CONFIGS = {
    "classification": {
        "script": "main_classification.py",
        "dataset": "imagenet",
        "data_env": "DAGA_IMAGENET_PATH",
        "input_size": 224,
        "metric_pattern": r"Best Acc:\s*([\d.]+)%",
        "direction": "maximize",
    },
    "detection": {
        "script": "main_detection.py",
        "dataset": "coco",
        "data_env": "DAGA_COCO_PATH",
        "input_size": 518,
        "metric_pattern": r"Final mAP:\s*([\d.]+)",
        "direction": "maximize",
        "out_indices_arg": "--layers_to_use",
    },
    "segmentation": {
        "script": "main_segmentation.py",
        "dataset": "ade20k",
        "data_env": "DAGA_ADE20K_PATH",
        "input_size": 518,
        "metric_pattern": r"Best mIoU:\s*([\d.]+)",
        "direction": "maximize",
        "out_indices_arg": "--out_indices",
    },
    "depth": {
        "script": "main_depth.py",
        "dataset": "nyu_depth_v2",
        "data_env": "DAGA_NYU_PATH",
        "input_size": 518,
        "metric_pattern": r"Best abs_rel:\s*([\d.]+)",
        "direction": "minimize",
        "out_indices_arg": "--out_indices",
    },
}

LAYER_CONFIGS = {
    "hourglass": [1, 2, 10, 11],
    "spread": [2, 5, 8, 11],
}


def run_experiment(task, config, trial_number, output_base):
    """Run experiment by directly calling main_*.py"""
    output_dir = f"{output_base}/trial_{trial_number:03d}"
    os.makedirs(output_dir, exist_ok=True)
    
    gpu_ids = os.environ.get("GPU_IDS", "0,1,2")
    num_gpus = len(gpu_ids.split(","))
    tc = TASK_CONFIGS[task]
    data_path = os.environ.get(tc["data_env"])
    if not data_path:
        raise ValueError(f"Set {tc['data_env']} to the dataset root before running {task} search")
    
    cmd = [
        "torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={num_gpus}",
        str(PROJECT_ROOT / tc["script"]),
        "--dataset", tc["dataset"],
        "--data_path", data_path,
        "--model_name", "dinov3_vitb16",
        "--pretrained_path", str(PROJECT_ROOT / "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        "--batch_size", str(get_batch_size(task)),
        "--input_size", str(tc["input_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--output_dir", output_dir,
        "--num_workers", "6",
        "--use_amp",
        "--use_daga",
        "--daga_layers", *[str(l) for l in config["daga_layers"]],
    ]
    
    if "out_indices_arg" in tc:
        cmd.extend([tc["out_indices_arg"], *[str(l) for l in config["daga_layers"]]])
    
    if "subset_ratio" in config and task != "depth":
        if task == "classification":
            cmd.extend(["--subset_ratio", str(config["subset_ratio"])])
        else:
            cmd.extend(["--sample_ratio", str(config["subset_ratio"])])
    
    log_file = f"{output_dir}/train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    layer_name = config.get('layer_config_name', 'custom')
    print(f"\n{'='*60}")
    print(f"Trial {trial_number}: LR={config['lr']:.6f}, Config={layer_name}, Layers={config['daga_layers']}")
    print(f"{'='*60}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(PROJECT_ROOT), text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
        process.wait()
    
    metric = None
    with open(log_file, "r") as f:
        content = f.read()
        match = re.search(tc["metric_pattern"], content)
        if match:
            metric = float(match.group(1))
    
    if metric is None:
        print(f"Warning: Could not parse metric from {log_file}")
        return 0.0 if tc["direction"] == "maximize" else float('inf')
    
    return metric


DEFAULT_LR_RANGES = {
    "classification": (0.1, 1.0),
    "detection": (0.0001, 0.1),
    "segmentation": (0.0001, 0.1),
    "depth": (0.0001, 0.1),
}

def get_lr_range(task):
    env_min = f"LR_MIN_{task.upper()}"
    env_max = f"LR_MAX_{task.upper()}"
    default_min, default_max = DEFAULT_LR_RANGES[task]
    return (
        float(os.environ.get(env_min, default_min)),
        float(os.environ.get(env_max, default_max))
    )

def get_subset_ratio(task, default):
    env_var = f"SUBSET_{task.upper()}"
    return float(os.environ.get(env_var, default))

def create_objective(task, output_base, epochs, subset_ratio, swanlab_run=None):
    def objective(trial):
        lr_min, lr_max = get_lr_range(task)
        actual_subset = get_subset_ratio(task, subset_ratio)
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        layer_config_name = trial.suggest_categorical("layer_config", ["hourglass", "spread"])
        daga_layers = LAYER_CONFIGS[layer_config_name]
        
        config = {
            "lr": lr,
            "daga_layers": daga_layers,
            "layer_config_name": layer_config_name,
            "epochs": epochs,
            "subset_ratio": actual_subset,
        }
        
        metric = run_experiment(task, config, trial.number, output_base)
        
        if swanlab_run is not None:
            layer_config_idx = 0 if layer_config_name == "hourglass" else 1
            swanlab.log({
                "trial": trial.number,
                "lr": lr,
                "layer_config_idx": layer_config_idx,
                "metric": metric,
            })
        
        return metric
    
    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search for DAGA")
    parser.add_argument("--task", type=str, required=True,
                       choices=["classification", "detection", "segmentation", "depth"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs/optuna_search")
    parser.add_argument("--enable_swanlab", action="store_true")
    
    args = parser.parse_args()
    
    output_base = f"{args.output_dir}/{args.task}"
    os.makedirs(output_base, exist_ok=True)
    
    tc = TASK_CONFIGS[args.task]
    
    swanlab_run = None
    if args.enable_swanlab and HAS_SWANLAB:
        swanlab_run = swanlab.init(
            project="DAGA-Optuna-Search",
            workspace="Dino_DAGA",
            experiment_name=f"optuna_{args.task}",
        )
        print("SwanLab logging enabled")
    
    study = optuna.create_study(
        study_name=f"daga_{args.task}",
        direction=tc["direction"],
    )
    
    objective = create_objective(args.task, output_base, args.epochs, args.subset_ratio, swanlab_run)
    
    print(f"\n{'='*60}")
    print(f"  Optuna Hyperparameter Search")
    print(f"  Task: {args.task}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs/trial: {args.epochs}")
    print(f"  Subset ratio: {args.subset_ratio}")
    print(f"{'='*60}\n")
    
    def save_callback(study, trial):
        results = {
            "task": args.task,
            "completed_trials": len(study.trials),
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_trial.value if study.best_trial else None,
            "best_params": study.best_trial.params if study.best_trial else None,
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in study.trials
            ]
        }
        with open(f"{output_base}/optuna_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True, callbacks=[save_callback])
    
    print(f"\n{'='*60}")
    print(f"  Best Trial")
    print(f"{'='*60}")
    
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    results = {
        "task": args.task,
        "best_trial": trial.number,
        "best_value": trial.value,
        "best_params": trial.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ]
    }
    
    results_file = f"{output_base}/optuna_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    if swanlab_run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
