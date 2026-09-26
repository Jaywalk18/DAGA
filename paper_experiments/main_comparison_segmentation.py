"""
Semantic Segmentation with multiple adaptation methods for paper comparison
Supports: baseline, daga, vit-adapter, adapter-former, lora, vpt-deep, layer-finetuning, full-finetune
"""
import os
import warnings
import logging

# Suppress DDP find_unused_parameters warning
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp, create_ddp_dataloaders
from core.daga import DAGA, process_attention_to_guidance
from data.segmentation_datasets import get_segmentation_dataset
from tasks.segmentation import setup_training_components, run_training_loop, prepare_visualization_data
from core.heads import LinearSegmentationHead

sys.path.insert(0, str(Path(__file__).parent / 'methods'))
from vit_adapter import ViTAdapter
from layer_finetuning import unfreeze_layers
from adapter_former import AdapterFormer
from lora import LoRA
from vpt import VPTDeep


class ComparisonSegmentationModel(nn.Module):
    """
    Segmentation model supporting multiple adaptation methods for paper comparison
    Methods: baseline, daga, vit-adapter, adapter-former, lora, vpt-deep, layer-finetuning, full-finetune
    """
    def __init__(
        self,
        pretrained_vit,
        num_classes=150,
        method='baseline',
        adaptation_layers=[2, 5, 8, 11],
        out_indices=[2, 5, 8, 11],
        **kwargs,
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.method = method
        self.adaptation_layers = adaptation_layers
        self.out_indices = out_indices
        self.feature_dim = self.vit.embed_dim
        self._kwargs = kwargs  # Store for method-specific params
        
        self.num_storage_tokens = -1
        self._cached_attn = None
        self._cached_guidance = None
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
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
        elif method == 'adapter-former':
            self.adapters = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                bottleneck_dim=64
            )
        elif method == 'lora':
            self.adapters = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=adaptation_layers,
                rank=8,
                alpha=16
            )
        elif method == 'vpt-deep':
            num_prompts = self._kwargs.get('num_prompts', 10)
            self.vpt = VPTDeep(
                embed_dim=self.feature_dim,
                num_prompts=num_prompts,
                vpt_layers=adaptation_layers
            )
        elif method == 'daga':
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1
            )
        elif method == 'layer-finetuning':
            unfreeze_layers(self.vit, adaptation_layers)
        elif method == 'full-finetune':
            for param in self.vit.parameters():
                param.requires_grad = True
        # baseline: backbone frozen, only head trained
        
        # Segmentation head using multi-scale features
        embed_dims = [self.feature_dim] * len(out_indices)
        self.decode_head = LinearSegmentationHead(
            in_channels=embed_dims,
            num_classes=num_classes
        )
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print trainable parameter count"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ ComparisonSegmentationModel [{self.method}]:")
        print(f"  - Total params: {total/1e6:.2f}M")
        print(f"  - Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)")
        print(f"  - Out indices: {self.out_indices}")
    
    def forward(self, x, request_visualization_maps=False):
        B = x.shape[0]
        input_size = (x.shape[2], x.shape[3])
        
        # Use ViT's prepare_tokens_with_masks method
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens < 0:
            self.num_storage_tokens = num_registers
        
        # Get guidance for DAGA (lightweight method)
        guidance = None
        if self.method == 'daga':
            guidance = self._get_guidance_map_lightweight(x_processed, H, W, num_registers)
        
        # Forward through blocks with adaptation
        multi_scale_features = []
        for i, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # VPT-Deep: insert prompts before block
            num_prompts_inserted = 0
            if self.method == 'vpt-deep' and i in self.adaptation_layers:
                x_processed, num_prompts_inserted = self.vpt.insert_prompts(x_processed, i)
            
            x_processed = block(x_processed, rope_sincos)
            
            # VPT-Deep: remove prompts after block
            if self.method == 'vpt-deep' and num_prompts_inserted > 0:
                x_processed = self.vpt.remove_prompts(x_processed, num_prompts_inserted)
            
            # Apply adaptation based on method (after block)
            if i in self.adaptation_layers:
                if self.method in ['vit-adapter', 'adapter-former', 'lora']:
                    x_processed = self.adapters.forward(x_processed, i)
                elif self.method == 'daga' and guidance is not None:
                    cls_token = x_processed[:, :1, :]
                    register_tokens = x_processed[:, 1:1+num_registers, :]
                    patch_tokens = x_processed[:, 1+num_registers:, :]
                    adapted_patch_tokens = self.daga.apply(patch_tokens, guidance, i, H, W)
                    x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            # Collect multi-scale features
            if i in self.out_indices:
                patch_features = x_processed[:, 1+num_registers:, :]
                feat_spatial = patch_features.transpose(1, 2).reshape(B, C, H, W)
                multi_scale_features.append(feat_spatial)
        
        # Segmentation head
        logits = self.decode_head(multi_scale_features, input_size)
        
        # Return format compatible with training loop: (logits, attn, guidance, baseline_attn)
        return logits, None, None, None
    
    def _get_guidance_map_lightweight(self, x_processed, H, W, num_registers):
        """Lightweight forward pass to get last layer's attention for DAGA guidance"""
        with torch.no_grad():
            features = x_processed.clone()
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                
                if idx == self.daga_guidance_layer_idx:
                    attn_weights = self._get_attention_inline(block, features)
                    guidance_map = process_attention_to_guidance(
                        attn_weights, H, W, num_registers
                    )
                    return guidance_map
                
                features = block(features, rope_sincos)
        return None
    
    def _get_attention_inline(self, block, x):
        """Compute attention weights inline without extra forward"""
        attn_module = block.attn
        normed_x = block.norm1(x)
        
        B, N, C = normed_x.shape
        num_heads = attn_module.num_heads
        head_dim = C // num_heads
        
        qkv = attn_module.qkv(normed_x).reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        return attn


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Segmentation Comparison Experiments")
    
    parser.add_argument("--method", type=str, required=True, 
                       choices=['baseline', 'daga', 'vit-adapter', 'adapter-former', 
                               'lora', 'vpt-deep', 'layer-finetuning', 'full-finetune'])
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts for VPT-Deep")
    
    # Model
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--out_indices", type=int, nargs="+", default=[2, 5, 8, 11])
    
    # Dataset
    parser.add_argument("--dataset", choices=["ade20k", "cityscapes", "voc", "pascal_voc"], default="ade20k")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    
    # Training
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=False)
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs")
    parser.add_argument("--enable_swanlab", action="store_true", default=False)
    parser.add_argument("--swanlab_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--enable_visualization", action="store_true", default=False)
    
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
        experiment_name = setup_logging(args, task_name="segmentation")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    train_dataset, val_dataset, num_classes = get_segmentation_dataset(args)
    
    train_loader, val_loader = create_ddp_dataloaders(
        train_dataset, val_dataset, args.batch_size, world_size, rank,
        num_workers=args.num_workers
    )
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # Create model with selected method
    model = ComparisonSegmentationModel(
        vit_model,
        num_classes=num_classes,
        method=args.method,
        adaptation_layers=args.adaptation_layers,
        out_indices=args.out_indices,
        num_prompts=args.num_prompts,
    )
    
    model.to(device)
    
    # Use find_unused_parameters for methods that may not use all params
    find_unused = args.method in ['full-finetune', 'layer-finetuning']
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=find_unused, broadcast_buffers=False,
                gradient_as_bucket_view=True)
    
    criterion, optimizer, scheduler = setup_training_components(model, args)
    
    best_miou, final_miou, total_time = run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args, output_dir, None, None, num_classes,
        rank=rank, world_size=world_size
    )
    
    if is_main_process:
        finalize_experiment(best_miou, final_miou, total_time, output_dir,
                          metric_name="mIoU", enable_swanlab=getattr(args, 'enable_swanlab', True))
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
