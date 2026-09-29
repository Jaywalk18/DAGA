"""
Depth estimation with multiple adaptation methods for paper comparison
Supports: Baseline, DAGA, ViT-Adapter, AdaptFormer, LoRA, Layer Fine-tuning
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp
from core.datasets import NYUDepthV2Dataset, KITTIDepthDataset

# Import comparison methods
sys.path.insert(0, str(Path(__file__).parent / 'methods'))
from vit_adapter import ViTAdapter
from layer_finetuning import unfreeze_layers
from adapter_former import AdapterFormer
from lora import LoRA

# Add dinov3 to path
dinov3_path = str(Path(__file__).parent.parent / 'dinov3')
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from dinov3.eval.depth.models import build_depther

warnings.filterwarnings("ignore")


class ComparisonDepthModel(nn.Module):
    """Depth estimation model supporting multiple adaptation methods"""
    
    def __init__(self, vit_model, method='baseline', adaptation_layers=None, 
                 out_indices=None, min_depth=0.001, max_depth=10.0,
                 guidance_dim=128, mlp_ratio=0.25, drop_rate=0.1):
        super().__init__()
        self.vit = vit_model
        self.method = method
        self.adaptation_layers = adaptation_layers or [2, 5, 8, 11]
        self.out_indices = out_indices or [2, 5, 8, 11]
        self.feature_dim = vit_model.embed_dim
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Freeze backbone initially
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize adaptation method
        if method == 'daga':
            from core.daga import DAGA, process_attention_to_guidance
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=guidance_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            )
            for param in self.daga.parameters():
                param.requires_grad = True
            # Use last layer for guidance
            self.daga_guidance_layer_idx = len(vit_model.blocks) - 1
            print(f"✓ Using DAGA on layers {adaptation_layers}")
            print(f"  - Guidance from layer {self.daga_guidance_layer_idx}")
            print(f"  - guidance_dim={guidance_dim}, mlp_ratio={mlp_ratio}, drop_rate={drop_rate}")
            
        elif method == 'vit-adapter':
            self.adapters = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                mlp_ratio=0.25
            )
            print(f"✓ Using ViT-Adapter on layers {adaptation_layers}")
        
        elif method == 'adapter-former':
            self.adapters = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                bottleneck_dim=64
            )
            print(f"✓ Using AdaptFormer on layers {adaptation_layers}")
        
        elif method == 'lora':
            self.adapters = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=adaptation_layers,
                rank=4
            )
            print(f"✓ Using LoRA on layers {adaptation_layers}")
            
        elif method == 'layer-finetuning':
            unfreeze_layers(self.vit, adaptation_layers)
            print(f"✓ Using Layer Fine-tuning on layers {adaptation_layers}")
            
        elif method == 'full-finetune':
            for param in self.vit.parameters():
                param.requires_grad = True
            print("✓ Using Full Fine-tuning")
        
        elif method == 'baseline':
            print("✓ Using Baseline (frozen backbone)")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build DPT depth head
        self._build_depth_head()
    
    def _build_depth_head(self):
        """Build DPT-style depth prediction head"""
        embed_dim = self.feature_dim
        
        # Feature fusion layers for multi-scale features
        self.reassemble = nn.ModuleList([
            nn.Conv2d(embed_dim, 256, kernel_size=1) for _ in self.out_indices
        ])
        
        # Fusion blocks
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(len(self.out_indices))
        ])
        
        # Final depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, _, H_img, W_img = x.shape
        
        # Get intermediate features
        features = self._forward_backbone(x)
        
        # Process multi-scale features
        H, W = features[0].shape[2], features[0].shape[3]
        
        fused = None
        for i, (feat, reassemble, fusion) in enumerate(zip(
            features, self.reassemble, self.fusion
        )):
            feat = reassemble(feat)
            feat = fusion(feat)
            
            if fused is None:
                fused = feat
            else:
                # Upsample and add
                fused = nn.functional.interpolate(fused, size=feat.shape[2:], mode='bilinear', align_corners=False)
                fused = fused + feat
        
        # Predict depth
        depth = self.depth_head(fused)
        depth = nn.functional.interpolate(depth, size=(H_img, W_img), mode='bilinear', align_corners=False)
        
        # Scale to depth range
        depth = self.min_depth + (self.max_depth - self.min_depth) * depth
        
        return depth
    
    def _forward_backbone(self, x):
        """Forward through backbone with adaptation"""
        from core.daga import process_attention_to_guidance
        
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        # Compute DAGA guidance if needed (using last layer attention)
        daga_guidance = None
        if self.method == 'daga':
            daga_guidance = self._get_daga_guidance(x_processed, H, W, num_registers)
        
        intermediate_features = []
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            x_processed = block(x_processed, rope_sincos)
            
            # Apply adaptation
            if idx in self.adaptation_layers:
                if self.method == 'daga' and daga_guidance is not None:
                    cls_token = x_processed[:, :1, :]
                    registers = x_processed[:, 1:1+num_registers, :]
                    patches = x_processed[:, 1+num_registers:, :]
                    # Use DAGA.apply() method correctly
                    patches = self.daga.apply(patches, daga_guidance, idx, H, W)
                    x_processed = torch.cat([cls_token, registers, patches], dim=1)
                    
                elif self.method in ['vit-adapter', 'adapter-former', 'lora']:
                    x_processed = self.adapters.forward(x_processed, idx)
            
            # Save intermediate features
            if idx in self.out_indices:
                patches = x_processed[:, 1+num_registers:, :]
                feat = patches.transpose(1, 2).reshape(-1, C, H, W)
                intermediate_features.append(feat)
        
        return intermediate_features
    
    def _get_daga_guidance(self, x_processed, H, W, num_registers):
        """Compute DAGA guidance map from last layer attention"""
        from core.daga import process_attention_to_guidance
        
        with torch.no_grad():
            features = x_processed.clone()
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                if idx == self.daga_guidance_layer_idx:
                    attn_module = block.attn
                    normed_x = block.norm1(features)
                    B, N, C = normed_x.shape
                    num_heads = attn_module.num_heads
                    head_dim = C // num_heads
                    qkv = attn_module.qkv(normed_x).reshape(B, N, 3, num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    q = q * attn_module.scale
                    attn = q @ k.transpose(-2, -1)
                    attn = attn.softmax(dim=-1)
                    return process_attention_to_guidance(attn, H, W, num_registers)
                features = block(features, rope_sincos)
        return None


def silog_loss(pred, target, mask=None, variance_focus=0.85):
    """Scale-Invariant Logarithmic Loss"""
    # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    if mask is None:
        mask = target > 0
    
    pred = pred[mask]
    target = target[mask]
    
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
    silog = torch.sqrt(torch.mean(log_diff ** 2) - variance_focus * (torch.mean(log_diff) ** 2))
    
    return silog


def compute_depth_metrics(pred, target, min_depth=0.001, max_depth=10.0):
    """Compute depth estimation metrics"""
    # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    mask = (target > min_depth) & (target < max_depth)
    
    pred = pred[mask]
    target = target[mask]
    
    if pred.numel() == 0:
        return {'rmse': 0, 'abs_rel': 0, 'delta1': 0}
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    
    # Delta accuracy (threshold = 1.25)
    thresh = torch.max(pred / target, target / pred)
    delta1 = (thresh < 1.25).float().mean() * 100
    
    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'delta1': delta1.item()
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Depth Comparison Experiments")
    
    parser.add_argument("--method", type=str, default="baseline",
                       choices=['baseline', 'daga', 'vit-adapter', 'adapter-former', 
                               'lora', 'layer-finetuning', 'full-finetune'])
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--out_indices", type=int, nargs="+", default=[2, 5, 8, 11])
    
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    
    parser.add_argument("--dataset", choices=["nyu_depth_v2", "kitti"], default="nyu_depth_v2")
    parser.add_argument("--data_path", type=str, required=True)
    
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true")
    
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=10.0)
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs/depth")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    
    # DAGA-specific hyperparameters
    parser.add_argument("--guidance_dim", type=int, default=128, help="DAGA guidance dimension")
    parser.add_argument("--mlp_ratio", type=float, default=0.25, help="DAGA MLP bottleneck ratio")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="DAGA dropout rate")
    
    # Save options
    parser.add_argument("--no_save_model", action="store_true", help="Don't save model checkpoint")
    
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
        print(f"Depth Comparison: {args.method.upper()}")
        print(f"Dataset: {args.dataset}")
        print(f"{'='*60}\n")
    
    # Load dataset
    if args.dataset == "nyu_depth_v2":
        train_dataset = NYUDepthV2Dataset(args.data_path, split='train', input_size=args.input_size)
        val_dataset = NYUDepthV2Dataset(args.data_path, split='val', input_size=args.input_size)
    else:
        train_dataset = KITTIDepthDataset(args.data_path, split='train', input_size=args.input_size)
        val_dataset = KITTIDepthDataset(args.data_path, split='val', input_size=args.input_size)
    
    # Subset if needed
    if args.sample_ratio < 1.0:
        n = int(len(train_dataset) * args.sample_ratio)
        train_dataset = torch.utils.data.Subset(train_dataset, range(n))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    if is_main:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load backbone and create model
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = ComparisonDepthModel(
        vit_model,
        method=args.method,
        adaptation_layers=args.adaptation_layers,
        out_indices=args.out_indices,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        guidance_dim=args.guidance_dim,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
    )
    model.to(device)
    # Use find_unused_parameters for full-finetune (storage_tokens not used)
    find_unused = args.method == 'full-finetune'
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=find_unused
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Training
    best_rmse = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_loss = 0
        for batch in tqdm(train_loader, disable=not is_main, desc=f"Epoch {epoch+1}"):
            # Dataset returns (image, depth) tuple
            images, depths = batch
            images = images.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred = model(images)
                loss = silog_loss(pred, depths)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        metrics_sum = {'rmse': 0, 'abs_rel': 0, 'delta1': 0}
        n_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Dataset returns (image, depth) tuple
                images, depths = batch
                images = images.to(device)
                depths = depths.to(device)
                
                pred = model(images)
                metrics = compute_depth_metrics(pred, depths, args.min_depth, args.max_depth)
                
                for k in metrics_sum:
                    metrics_sum[k] += metrics[k] * images.size(0)
                n_samples += images.size(0)
        
        # Gather metrics
        for k in metrics_sum:
            tensor = torch.tensor([metrics_sum[k], n_samples], device=device)
            dist.all_reduce(tensor)
            metrics_sum[k] = tensor[0].item() / tensor[1].item()
        
        if is_main:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
                  f"RMSE={metrics_sum['rmse']:.4f}, δ1={metrics_sum['delta1']:.2f}%")
            
            if metrics_sum['rmse'] < best_rmse:
                best_rmse = metrics_sum['rmse']
                if not args.no_save_model:
                    torch.save(model.state_dict(), output_dir / 'best_model.pth')
    
    if is_main:
        print(f"\nBest RMSE: {best_rmse:.4f}")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

