"""
KNN evaluation with multiple adaptation methods for paper comparison
Supports: Baseline, DAGA, ViT-Adapter, AdaptFormer, LoRA
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone, compute_daga_guidance_map
from core.utils import setup_environment
from core.ddp_utils import setup_ddp, cleanup_ddp
from data.classification_datasets import get_classification_dataset
from core.daga import DAGA

# Import comparison methods
sys.path.insert(0, str(Path(__file__).parent / 'methods'))
from vit_adapter import ViTAdapter
from adapter_former import AdapterFormer
from lora import LoRA

warnings.filterwarnings("ignore")


class FeatureExtractor(nn.Module):
    """Feature extractor supporting multiple adaptation methods"""
    
    def __init__(self, vit_model, method='baseline', adaptation_layers=None):
        super().__init__()
        self.vit = vit_model
        self.method = method
        self.adaptation_layers = adaptation_layers or [1, 2, 10, 11]
        self.feature_dim = vit_model.embed_dim
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize adaptation method
        if method == 'daga':
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in adaptation_layers
            })
            print(f"✓ Using DAGA on layers {adaptation_layers}")
            
        elif method == 'vit-adapter':
            self.adapters = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers
            )
            print(f"✓ Using ViT-Adapter")
        
        elif method == 'adapter-former':
            self.adapters = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers
            )
            print(f"✓ Using AdaptFormer")
        
        elif method == 'lora':
            self.adapters = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=adaptation_layers
            )
            print(f"✓ Using LoRA")
        
        elif method == 'baseline':
            print("✓ Using Baseline (frozen)")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @torch.no_grad()
    def forward(self, x):
        """Extract CLS token features"""
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        # Compute guidance for DAGA
        daga_guidance = None
        if self.method == 'daga':
            guidance_layer = max(self.adaptation_layers)
            daga_guidance = compute_daga_guidance_map(
                self.vit, x_processed, H, W, guidance_layer
            )
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            x_processed = block(x_processed, rope_sincos)
            
            if idx in self.adaptation_layers:
                if self.method == 'daga' and daga_guidance is not None:
                    cls = x_processed[:, :1, :]
                    regs = x_processed[:, 1:1+num_registers, :]
                    patches = x_processed[:, 1+num_registers:, :]
                    patches = self.daga_modules[str(idx)](patches, daga_guidance)
                    x_processed = torch.cat([cls, regs, patches], dim=1)
                    
                elif self.method in ['vit-adapter', 'adapter-former', 'lora']:
                    x_processed = self.adapters.forward(x_processed, idx)
        
        x_norm = self.vit.norm(x_processed)
        return x_norm[:, 0]  # CLS token


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract all features from dataloader"""
    model.eval()
    features_list = []
    labels_list = []
    
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model(images)
        features = nn.functional.normalize(features, dim=1)
        features_list.append(features.cpu())
        labels_list.append(labels)
    
    return torch.cat(features_list), torch.cat(labels_list)


def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, temperature=0.07):
    """KNN classification"""
    num_classes = train_labels.max().item() + 1
    
    # Compute similarities
    sim = test_features @ train_features.T / temperature
    
    # Get top-k
    topk_sim, topk_idx = sim.topk(k, dim=1)
    topk_labels = train_labels[topk_idx]
    
    # Weight by similarity
    weights = torch.softmax(topk_sim, dim=1)
    
    # Vote
    pred = torch.zeros(test_features.size(0), num_classes)
    for c in range(num_classes):
        mask = (topk_labels == c).float()
        pred[:, c] = (weights * mask).sum(dim=1)
    
    predicted = pred.argmax(dim=1)
    correct = (predicted == test_labels).sum().item()
    
    return correct / len(test_labels) * 100


def parse_arguments():
    parser = argparse.ArgumentParser(description="KNN Comparison")
    
    parser.add_argument("--method", type=str, default="baseline",
                       choices=['baseline', 'daga', 'vit-adapter', 'adapter-former', 'lora'])
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[1, 2, 10, 11])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_path", type=str, required=True)
    
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--knn_k_values", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--temperature", type=float, default=0.07)
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs/knn")
    
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
        print(f"KNN Evaluation: {args.method.upper()}")
        print(f"Dataset: {args.dataset}")
        print(f"K values: {args.knn_k_values}")
        print(f"{'='*60}\n")
    
    # Load dataset
    train_dataset, test_dataset, num_classes = get_classification_dataset(args)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Load model
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    model = FeatureExtractor(vit_model, method=args.method, adaptation_layers=args.adaptation_layers)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded checkpoint from {args.checkpoint}")
    
    model.to(device)
    
    # Extract features
    if is_main:
        print("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    if is_main:
        print("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # KNN evaluation
    if is_main:
        print("\nKNN Results:")
        print("-" * 40)
        
        results = {}
        for k in args.knn_k_values:
            acc = knn_classifier(train_features, train_labels, test_features, test_labels, 
                               k=k, temperature=args.temperature)
            results[f'k={k}'] = acc
            print(f"  K={k:3d}: {acc:.2f}%")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / f'{args.method}_results.json', 'w') as f:
            json.dump({
                'method': args.method,
                'dataset': args.dataset,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

