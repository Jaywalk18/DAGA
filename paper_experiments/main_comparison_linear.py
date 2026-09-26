"""
Linear probing evaluation with multiple adaptation methods for paper comparison
Supports: Baseline, DAGA
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone, compute_daga_guidance_map
from core.utils import setup_environment
from core.ddp_utils import setup_ddp, cleanup_ddp
from data.classification_datasets import get_classification_dataset
from core.daga import DAGA

warnings.filterwarnings("ignore")


class FeatureExtractor(nn.Module):
    """Feature extractor with optional DAGA"""
    
    def __init__(self, vit_model, use_daga=False, daga_layers=None):
        super().__init__()
        self.vit = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers or [1, 2, 10, 11]
        self.feature_dim = vit_model.embed_dim
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if use_daga:
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers
            })
    
    @torch.no_grad()
    def forward(self, x):
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        daga_guidance = None
        if self.use_daga:
            guidance_layer = max(self.daga_layers)
            daga_guidance = compute_daga_guidance_map(
                self.vit, x_processed, H, W, guidance_layer
            )
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            x_processed = block(x_processed, rope_sincos)
            
            if self.use_daga and idx in self.daga_layers and daga_guidance is not None:
                cls = x_processed[:, :1, :]
                regs = x_processed[:, 1:1+num_registers, :]
                patches = x_processed[:, 1+num_registers:, :]
                patches = self.daga_modules[str(idx)](patches, daga_guidance)
                x_processed = torch.cat([cls, regs, patches], dim=1)
        
        x_norm = self.vit.norm(x_processed)
        return nn.functional.normalize(x_norm[:, 0], dim=1)


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features_list, labels_list = [], []
    
    for images, labels in tqdm(dataloader, desc="Extracting"):
        features = model(images.to(device))
        features_list.append(features.cpu())
        labels_list.append(labels)
    
    return torch.cat(features_list), torch.cat(labels_list)


def train_linear_classifier(train_features, train_labels, test_features, test_labels,
                           num_classes, lr=1e-3, epochs=10, batch_size=256, device='cuda'):
    """Train linear classifier on extracted features"""
    classifier = nn.Linear(train_features.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    classifier.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_features.to(device))
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == test_labels).float().mean().item() * 100
    
    return acc


def parse_arguments():
    parser = argparse.ArgumentParser(description="Linear Probe Comparison")
    
    parser.add_argument("--use_daga", action="store_true")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[1, 2, 10, 11])
    parser.add_argument("--checkpoint", type=str, default=None)
    
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_path", type=str, required=True)
    
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--linear_epochs", type=int, default=10)
    parser.add_argument("--learning_rates", type=float, nargs="+", 
                       default=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs/linear")
    
    return parser.parse_args()


def main():
    local_rank, rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    method = "DAGA" if args.use_daga else "Baseline"
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Linear Probe: {method}")
        print(f"Dataset: {args.dataset}")
        print(f"LRs to test: {args.learning_rates}")
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
    model = FeatureExtractor(vit_model, use_daga=args.use_daga, daga_layers=args.daga_layers)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded checkpoint")
    
    model.to(device)
    
    # Extract features
    if is_main:
        print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    if is_main:
        print(f"Train: {train_features.shape}, Test: {test_features.shape}")
    
    # Train linear probes with different LRs
    if is_main:
        print("\nLinear Probe Results:")
        print("-" * 40)
        
        results = {}
        best_acc, best_lr = 0, 0
        
        for lr in args.learning_rates:
            acc = train_linear_classifier(
                train_features, train_labels, test_features, test_labels,
                num_classes, lr=lr, epochs=args.linear_epochs, device=device
            )
            results[f'lr={lr}'] = acc
            print(f"  LR={lr:.0e}: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc, best_lr = acc, lr
        
        print(f"\nBest: LR={best_lr:.0e}, Acc={best_acc:.2f}%")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / f'{method.lower()}_results.json', 'w') as f:
            json.dump({
                'method': method,
                'dataset': args.dataset,
                'best_lr': best_lr,
                'best_acc': best_acc,
                'all_results': results
            }, f, indent=2)
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

