"""
Robustness evaluation on ImageNet-C with multiple adaptation methods
Supports: Baseline, DAGA, LoRA, VPT-Deep, Adapter-Former, ViT-Adapter, Layer-Finetuning, Full-Finetune
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone, compute_daga_guidance_map
from core.utils import setup_environment
from core.ddp_utils import setup_ddp, cleanup_ddp
from core.daga import DAGA

# Import adapters from paper_experiments/methods
from paper_experiments.methods.lora import LoRA
from paper_experiments.methods.vpt import VPTDeep
from paper_experiments.methods.adapter_former import AdapterFormer
from paper_experiments.methods.vit_adapter import ViTAdapter

from torchvision import transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")

# ImageNet-C corruption types
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


class RobustnessModel(nn.Module):
    """Classification model for robustness evaluation with multiple adaptation methods"""
    
    def __init__(self, vit_model, method, num_classes=1000, 
                 adaptation_layers=None, num_prompts=10):
        super().__init__()
        self.vit = vit_model
        self.method = method
        self.adaptation_layers = adaptation_layers or [1, 2, 10, 11]
        self.feature_dim = vit_model.embed_dim
        self.num_prompts = num_prompts
        
        # Freeze backbone by default
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize adapters based on method
        self._init_adapters()
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def _init_adapters(self):
        """Initialize adapters based on method"""
        if self.method == 'daga':
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=self.adaptation_layers
            )
        elif self.method == 'lora':
            self.adapters = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=self.adaptation_layers,
                rank=4, alpha=1.0  # Must match training config
            )
        elif self.method == 'vpt-deep':
            self.vpt = VPTDeep(
                embed_dim=self.feature_dim,
                num_prompts=self.num_prompts,
                vpt_layers=self.adaptation_layers
            )
        elif self.method == 'adapter-former':
            self.adapters = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=self.adaptation_layers,
                bottleneck_dim=64
            )
        elif self.method == 'vit-adapter':
            self.adapters = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=self.adaptation_layers,
                mlp_ratio=0.25
            )
        elif self.method == 'layer-finetuning':
            # Unfreeze specific layers
            for idx in self.adaptation_layers:
                if idx < len(self.vit.blocks):
                    for param in self.vit.blocks[idx].parameters():
                        param.requires_grad = True
        elif self.method == 'full-finetune':
            for param in self.vit.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        # DAGA guidance
        daga_guidance = None
        if self.method == 'daga':
            guidance_layer = max(self.adaptation_layers)
            daga_guidance = compute_daga_guidance_map(
                self.vit, x_processed, H, W, guidance_layer
            )
        
        # VPT: will insert prompts per layer
        num_prompts_inserted = 0
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # VPT-Deep: insert prompts before block
            if self.method == 'vpt-deep' and idx in self.adaptation_layers:
                x_processed, num_prompts_inserted = self.vpt.insert_prompts(x_processed, idx)
            
            # Forward through block
            x_processed = block(x_processed, rope_sincos)
            
            # VPT-Deep: remove prompts after block
            if self.method == 'vpt-deep' and num_prompts_inserted > 0:
                x_processed = self.vpt.remove_prompts(x_processed, num_prompts_inserted)
                num_prompts_inserted = 0
            
            # Apply adaptation based on method
            if idx in self.adaptation_layers:
                if self.method == 'daga' and daga_guidance is not None:
                    cls = x_processed[:, :1, :]
                    regs = x_processed[:, 1:1+num_registers, :]
                    patches = x_processed[:, 1+num_registers:, :]
                    adapted_patches = self.daga.apply(patches, daga_guidance, idx, H, W)
                    x_processed = torch.cat([cls, regs, adapted_patches], dim=1)
                elif self.method == 'lora':
                    x_processed = self.adapters.forward(x_processed, idx)
                elif self.method == 'adapter-former':
                    x_processed = self.adapters.forward(x_processed, idx)
                elif self.method == 'vit-adapter':
                    x_processed = self.adapters.forward(x_processed, idx)
        
        x_norm = self.vit.norm(x_processed)
        features = x_norm[:, 0]
        return self.classifier(features)


def evaluate_corruption(model, data_path, corruption, severity, args, device):
    """Evaluate on a specific corruption type and severity"""
    corruption_path = os.path.join(data_path, corruption, str(severity))
    
    if not os.path.exists(corruption_path):
        return None
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(corruption_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total * 100 if total > 0 else 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Robustness Comparison")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=['baseline', 'daga', 'lora', 'vpt-deep', 
                               'adapter-former', 'vit-adapter', 'layer-finetuning', 'full-finetune'])
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[1, 2, 10, 11])
    parser.add_argument("--checkpoint", type=str, required=True)
    
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    
    parser.add_argument("--data_path", type=str, required=True)
    
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--corruption_types", type=str, nargs="+", default=CORRUPTION_TYPES)
    parser.add_argument("--severity_levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs/robustness")
    
    return parser.parse_args()


def main():
    local_rank, rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Robustness Evaluation: {args.method}")
        print(f"Corruption types: {len(args.corruption_types)}")
        print(f"Severity levels: {args.severity_levels}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"{'='*60}\n")
    
    # Load model
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    model = RobustnessModel(
        vit_model, 
        method=args.method,
        num_classes=1000,
        adaptation_layers=args.adaptation_layers
    )
    
    # Load trained checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if is_main:
            print(f"✓ Loaded checkpoint from {args.checkpoint}")
    else:
        if is_main:
            print(f"⚠️ Warning: Checkpoint not found: {args.checkpoint}")
    
    model.to(device)
    
    # Evaluate on each corruption
    results = {}
    
    if is_main:
        print("\nEvaluating corruptions...")
        print("-" * 60)
    
    for corruption in tqdm(args.corruption_types, disable=not is_main, desc="Corruptions"):
        corruption_accs = []
        
        for severity in args.severity_levels:
            acc = evaluate_corruption(model, args.data_path, corruption, severity, args, device)
            if acc is not None:
                corruption_accs.append(acc)
        
        if corruption_accs:
            mean_acc = sum(corruption_accs) / len(corruption_accs)
            results[corruption] = {
                'mean': mean_acc,
                'by_severity': {str(s): a for s, a in zip(args.severity_levels, corruption_accs)}
            }
            
            if is_main:
                print(f"  {corruption:20s}: {mean_acc:.2f}%")
    
    # Compute overall mean
    if results:
        overall_mean = sum(r['mean'] for r in results.values()) / len(results)
        results['overall_mean'] = overall_mean
        
        if is_main:
            print("-" * 60)
            print(f"  {'Overall Mean':20s}: {overall_mean:.2f}%")
            
            # Save results
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / f'{args.method}_results.json', 'w') as f:
                json.dump({
                    'method': args.method,
                    'checkpoint': args.checkpoint,
                    'results': results
                }, f, indent=2)
            
            print(f"\n✓ Results saved to {output_dir}/{args.method}_results.json")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
