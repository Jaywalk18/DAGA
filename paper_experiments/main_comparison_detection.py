"""
Object Detection with multiple adaptation methods for paper comparison
Supports: baseline, daga, daga-plus, vit-adapter, adapter-former, lora, vpt-deep, layer-finetuning
"""
import os
import warnings
import logging

# Suppress DDP find_unused_parameters warning at multiple levels
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp, create_ddp_dataloaders
from data.detection_datasets import get_detection_dataset, detection_collate_fn
from tasks.detection import (
    setup_training_components, run_training_loop, prepare_visualization_data
)
from core.simple_detection_head import SimpleDetectionHead
from core.daga import DAGA, process_attention_to_guidance, AttentionHook

sys.path.insert(0, str(Path(__file__).parent / 'methods'))
from vit_adapter import ViTAdapter
from layer_finetuning import unfreeze_layers
from adapter_former import AdapterFormer
from lora import LoRA
from vpt import VPTDeep


class ComparisonDetectionModel(nn.Module):
    """
    Detection model supporting multiple adaptation methods for paper comparison
    Methods: baseline, daga, daga-lite, daga-det,
             vit-adapter, adapter-former, lora, vpt-deep, layer-finetuning
    """
    def __init__(
        self,
        pretrained_vit,
        num_classes=91,
        method='baseline',
        adaptation_layers=[2, 5, 8, 11],
        layers_to_use=[2, 5, 8, 11],
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.method = method
        self.adaptation_layers = adaptation_layers
        self.layers_to_use = layers_to_use
        self.feature_dim = self.vit.embed_dim
        
        self.num_storage_tokens = -1
        self._cached_attn = None
        self._cached_guidance = None
        
        # Feature stride (depends on ViT patch size)
        self.stride = self.vit.patch_size
        
        # Freeze backbone initially
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize adaptation modules based on method
        if method == 'vit-adapter':
            self.adapters = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                mlp_ratio=0.25
            )
            for param in self.adapters.parameters():
                param.requires_grad = True
            print(f"✓ Using ViT-Adapter on layers {adaptation_layers}")
        
        elif method == 'adapter-former':
            self.adapters = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                bottleneck_dim=64
            )
            for param in self.adapters.parameters():
                param.requires_grad = True
            print(f"✓ Using AdapterFormer on layers {adaptation_layers}")
        
        elif method == 'lora':
            self.adapters = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=adaptation_layers,
                rank=4,
                alpha=1.0
            )
            for param in self.adapters.parameters():
                param.requires_grad = True
            print(f"✓ Using LoRA on layers {adaptation_layers}")
        
        elif method == 'vpt-deep':
            self.vpt = VPTDeep(
                embed_dim=self.feature_dim,
                num_prompts=10,
                vpt_layers=adaptation_layers
            )
            for param in self.vpt.parameters():
                param.requires_grad = True
            print(f"✓ Using VPT-Deep on layers {adaptation_layers}")
            
        elif method == 'layer-finetuning':
            trainable_params = unfreeze_layers(self.vit, adaptation_layers)
            print(f"✓ Using Layer Fine-tuning on layers {adaptation_layers}")
            print(f"  Trainable params: {trainable_params/1e6:.2f}M")
        
        elif method == 'daga':
            # DAGA: shared extractor with layer-specific adapters
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
            for param in self.daga.parameters():
                param.requires_grad = True
            # Use LAST layer for guidance (delayed mode: use previous batch's guidance)
            self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
            self.is_shared_encoder = True
            self.use_single_pass = True
            self.use_delayed_guidance = True  # Use previous batch's guidance
            # Cache for delayed guidance
            self._prev_guidance = None
            params = self.daga.get_param_count()
            print(f"✓ Using DAGA on layers {adaptation_layers}")
            print(f"  - Guidance from layer {self.daga_guidance_layer_idx} (delayed)")
            print(f"  - Params: {params['total']:,}")
        
        elif method == 'daga-plus':
            # DAGA: Shared extractor + learnable spatial scale
            self.daga_plus = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
            for param in self.daga_plus.parameters():
                param.requires_grad = True
            self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
            self.is_shared_encoder = True
            self.use_single_pass = False
            self.use_delayed_guidance = True
            self._prev_guidance = None
            params = self.daga_plus.get_param_count()
            print(f"✓ Using DAGA on layers {adaptation_layers}")
            print(f"  - Params: {params['total']:,}")

        elif method == 'full-finetune':
            # Unfreeze ALL backbone parameters
            for param in self.vit.parameters():
                param.requires_grad = True
            total_params = sum(p.numel() for p in self.vit.parameters())
            print(f"✓ Using Full Fine-tuning (all backbone unfrozen)")
            print(f"  Trainable backbone params: {total_params/1e6:.2f}M")
            self.is_shared_encoder = False

        elif method == 'baseline':
            print("✓ Using Baseline (frozen backbone)")
            self.is_shared_encoder = False
        else:
            raise ValueError(f"Unknown method: {method}")

        # 确保非DAGA方法也有is_shared_encoder属性
        if not hasattr(self, 'is_shared_encoder'):
            self.is_shared_encoder = False
        
        # Detection head
        total_feature_dim = self.feature_dim * len(layers_to_use)
        self.detection_head = SimpleDetectionHead(total_feature_dim, num_classes)
        for param in self.detection_head.parameters():
            param.requires_grad = True
        
        print(
            f"✓ ComparisonDetectionModel initialized:\n"
            f"  - Method: {method}\n"
            f"  - Feature dim: {self.feature_dim} x {len(layers_to_use)} = {total_feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Adaptation layers: {adaptation_layers}\n"
            f"  - Feature extraction layers: {layers_to_use}"
        )
    
    def _get_daga_guidance(self, x_processed, H, W, num_registers):
        """Compute DAGA guidance map from last layer attention (CLS-based)"""
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

    def _get_daga_guidance_detection(self, x_processed, H, W, num_registers):
        """Compute detection-specific guidance using patch entropy"""
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
                    return detection_attention_guidance(attn, H, W, num_registers)
                features = block(features, rope_sincos)
        return None

    def forward(self, x, request_visualization_maps=False):
        """Forward pass with selected adaptation method"""
        if isinstance(x, list):
            x = torch.stack(x)
        
        B = x.shape[0]
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        # DAGA guidance setup
        daga_guidance_map = None
        current_guidance = None  # Will be captured this batch for next batch
        daga_methods = ['daga', 'daga-plus']
        use_single_pass = getattr(self, 'use_single_pass', False) and self.method in daga_methods
        use_delayed_guidance = getattr(self, 'use_delayed_guidance', False)
        
        # Delayed guidance mode: use PREVIOUS batch's guidance
        if use_delayed_guidance and self.method in daga_methods:
            daga_guidance_map = getattr(self, '_prev_guidance', None)
            # Fallback: use uniform guidance if None OR batch size mismatch
            if daga_guidance_map is None or daga_guidance_map.shape[0] != B:
                    daga_guidance_map = torch.ones(B, H, W, device=x.device) * 0.5
            
            self._cached_guidance = daga_guidance_map
        
        intermediate_features = []
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # VPT-Deep: insert prompts before block
            num_prompts_inserted = 0
            if self.method == 'vpt-deep' and idx in self.adaptation_layers:
                x_processed, num_prompts_inserted = self.vpt.insert_prompts(x_processed, idx)
            
            x_processed = block(x_processed, rope_sincos)
            
            # VPT-Deep: remove prompts after block
            if self.method == 'vpt-deep' and num_prompts_inserted > 0:
                x_processed = self.vpt.remove_prompts(x_processed, num_prompts_inserted)
            
            # Apply adaptation based on method (using PREVIOUS batch's guidance)
            if idx in self.adaptation_layers:
                if self.method in ['vit-adapter', 'adapter-former', 'lora']:
                    x_processed = self.adapters.forward(x_processed, idx)
                elif self.method == 'daga':
                    # DAGA: shared extractor with layer-specific adapters
                    cls_token = x_processed[:, :1, :]
                    register_tokens = x_processed[:, 1:1+num_registers, :]
                    patch_tokens = x_processed[:, 1+num_registers:, :]
                    adapted_patch_tokens = self.daga.apply(patch_tokens, daga_guidance_map, idx, H, W)
                    x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                elif self.method == 'daga-plus':
                    # DAGA-Plus: dual-path gate (channel + spatial), per-layer encoding
                    cls_token = x_processed[:, :1, :]
                    register_tokens = x_processed[:, 1:1+num_registers, :]
                    patch_tokens = x_processed[:, 1+num_registers:, :]
                    adapted_patch_tokens = self.daga_plus.apply(patch_tokens, daga_guidance_map, idx, H, W)
                    x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            # Store intermediate features for specified layers
            if idx in self.layers_to_use:
                patch_tokens = x_processed[:, 1 + num_registers:, :]
                feat_spatial = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                intermediate_features.append(feat_spatial)
        
        # Update cache: compute THIS batch's attention -> NEXT batch's guidance
        # For DAGA-Plus: compute CLS-to-patches attention
        if use_delayed_guidance and self.method == 'daga-plus':
            with torch.no_grad():
                last_block = self.vit.blocks[self.daga_guidance_layer_idx]
                attn_module = last_block.attn
                normed_x = last_block.norm1(x_processed)
                B_curr, N_curr, C_curr = normed_x.shape
                num_heads = attn_module.num_heads
                head_dim = C_curr // num_heads
                
                # Only compute CLS query and all keys (not full NxN attention)
                qkv = attn_module.qkv(normed_x).reshape(B_curr, N_curr, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                q, k, _ = qkv.unbind(0)
                
                # CLS query only: shape (B, heads, 1, head_dim)
                q_cls = q[:, :, 0:1, :] * attn_module.scale
                # All keys: shape (B, heads, N, head_dim)
                
                # CLS attention to all tokens: (B, heads, 1, N) -> (B, heads, N)
                cls_attn = (q_cls @ k.transpose(-2, -1)).softmax(dim=-1).squeeze(2)
                
                # Extract attention to patches only (skip CLS and registers)
                patch_start = 1 + num_registers
                cls_attn_patches = cls_attn[:, :, patch_start:]  # (B, heads, H*W)
                
                # Average over heads and reshape
                cls_attn_avg = cls_attn_patches.mean(dim=1)  # (B, H*W)
                
                # Normalize to [0, 1]
                min_val = cls_attn_avg.amin(dim=1, keepdim=True)
                max_val = cls_attn_avg.amax(dim=1, keepdim=True)
                cls_attn_norm = (cls_attn_avg - min_val) / (max_val - min_val + 1e-8)
                
                current_guidance = cls_attn_norm.reshape(B_curr, H, W)
        
        if use_delayed_guidance and current_guidance is not None:
            self._prev_guidance = current_guidance.detach()
        
        features_concat = torch.cat(intermediate_features, dim=1)
        
        cls_logits, box_preds, centerness = self.detection_head(features_concat)
        
        if request_visualization_maps:
            return cls_logits, box_preds, centerness, self._cached_attn, self._cached_guidance, None
        else:
            return cls_logits, box_preds, centerness


def setup_layer_finetuning_optimizer(model, args):
    """Setup optimizer for layer-finetuning with lower LR for backbone layers."""
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    backbone_params = []
    head_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "detection_head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    backbone_lr = 1e-5
    head_lr = args.lr
    
    param_groups = [
        {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    
    warmup_epochs = max(1, args.epochs // 20)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"✓ Layer-finetuning optimizer:")
    print(f"  Head params: {sum(p.numel() for p in head_params):,}, LR={head_lr}")
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}, LR={backbone_lr}")
    
    return optimizer, scheduler


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Detection Comparison Experiments")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=['baseline', 'full-finetune', 'daga', 'daga-plus', 'vit-adapter', 'adapter-former', 'lora', 'vpt-deep', 'layer-finetuning'])
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[1, 2, 10, 11])
    
    # Model
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--layers_to_use", type=int, nargs="+", default=[1, 2, 10, 11])
    
    # Dataset
    parser.add_argument("--dataset", choices=["coco"], default="coco")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    
    # Training
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs")
    parser.add_argument("--enable_swanlab", action="store_true", default=True)
    parser.add_argument("--swanlab_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--enable_visualization", action="store_true")
    parser.add_argument("--num_vis_samples", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Enable mixed precision training")
    parser.add_argument("--use_compile", action="store_true", default=False,
                       help="Enable torch.compile() (PyTorch 2.0+)")
    
    return parser.parse_args()


def main():
    from torch.nn.parallel import DistributedDataParallel as DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        args.swanlab_name = f"{args.dataset}_{args.method}_L{'-'.join(map(str, args.adaptation_layers))}"
        experiment_name = setup_logging(args, task_name="detection_comparison")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Paper Comparison: {args.method.upper()} - Detection")
        print(f"DDP Training with {world_size} GPUs")
        print(f"Adaptation layers: {args.adaptation_layers}")
    
    train_dataset, val_dataset, num_classes = get_detection_dataset(args)
    
    if is_main_process:
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    train_loader, val_loader = create_ddp_dataloaders(
        train_dataset, val_dataset, args.batch_size, world_size, rank,
        num_workers=args.num_workers, collate_fn=detection_collate_fn
    )
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = ComparisonDetectionModel(
        vit_model,
        num_classes=num_classes,
        method=args.method,
        adaptation_layers=args.adaptation_layers,
        layers_to_use=args.layers_to_use,
    )
    
    model.to(device)
    
    # Optional: torch.compile for optimization (PyTorch 2.0+)
    if args.use_compile:
        if is_main_process:
            print("[OK] Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            if is_main_process:
                print(f"[WARN] torch.compile failed: {e}, continuing without compilation")
    
    # All params used in forward pass (fusion gets gradients via zero-weight contribution)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, broadcast_buffers=False,
                gradient_as_bucket_view=True)
    
    if is_main_process:
        print(f"✓ Model wrapped with DDP on {world_size} GPUs")
        print(f"✓ Optimizations: AMP={args.use_amp}, Compile={args.use_compile}\n")
    
    # Prepare visualization
    fixed_vis_images, fixed_vis_boxes = None, None
    if is_main_process and args.enable_visualization:
        fixed_vis_images, fixed_vis_boxes = prepare_visualization_data(val_dataset, args, device)
    
    # Setup optimizer
    if args.method == 'layer-finetuning':
        optimizer, scheduler = setup_layer_finetuning_optimizer(model, args)
    else:
        optimizer, scheduler = setup_training_components(model, args)
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Starting training with {args.method}")
        print(f"{'='*70}\n")
    
    best_loss, final_metrics, total_time = run_training_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        device, args, output_dir, fixed_vis_images, fixed_vis_boxes, num_classes,
        rank=rank, world_size=world_size, use_amp=getattr(args, 'use_amp', False)
    )

    if is_main_process:
        # Save results to local CSV/JSON for offline analysis
        import json
        import pandas as pd

        results = {
            'method': args.method,
            'adaptation_layers': args.adaptation_layers,
            'layers_to_use': args.layers_to_use,
            'final_mAP': final_metrics.get('mAP', 0),
            'final_mAP50': final_metrics.get('mAP@50', 0),
            'final_mAP75': final_metrics.get('mAP@75', 0),
            'best_loss': best_loss,
            'total_time_min': total_time / 60,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size * world_size,
        }

        # Save JSON
        results_json_path = output_dir / 'results_summary.json'
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_json_path}")

        # Save CSV (append mode for easy comparison)
        results_csv_path = Path(args.output_dir) / 'all_detection_results.csv'
        df = pd.DataFrame([results])
        if results_csv_path.exists():
            df_existing = pd.read_csv(results_csv_path)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(results_csv_path, index=False)
        print(f"✓ Results appended to: {results_csv_path}")

        finalize_experiment(
            best_loss, final_metrics.get('mAP', 0), total_time, output_dir,
            metric_name="mAP", enable_swanlab=getattr(args, 'enable_swanlab', True)
        )
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
