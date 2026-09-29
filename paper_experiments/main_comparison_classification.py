"""
Classification training with multiple adaptation methods for paper comparison
Supports: DAGA, ViT-Adapter, Layer Fine-tuning
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp, create_ddp_dataloaders
from data.classification_datasets import get_classification_dataset
from tasks.classification import ClassificationModel, setup_training_components, run_training_loop, prepare_visualization_data

# Import comparison methods
sys.path.insert(0, str(Path(__file__).parent / 'methods'))
from vit_adapter import ViTAdapter
from layer_finetuning import unfreeze_layers, get_layer_lr_groups
from adapter_former import AdapterFormer
from lora import LoRA
from vpt import VPTDeep


class ComparisonClassificationModel(nn.Module):
    """
    Classification model supporting multiple adaptation methods
    Methods: baseline, daga, vit-adapter, adapter-former, layer-finetuning
    """
    def __init__(
        self,
        pretrained_vit,
        num_classes=10,
        method='baseline',  # 'baseline', 'daga', 'vit-adapter', 'adapter-former', 'layer-finetuning'
        adaptation_layers=[11],
        enable_visualization=False,
        vis_attn_layer=11,
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.method = method
        self.adaptation_layers = adaptation_layers
        self.feature_dim = self.vit.embed_dim
        self.enable_visualization = enable_visualization
        self.vis_attn_layer = vis_attn_layer
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        self.captured_attn = None
        self.captured_guidance_attn = None
        self._attention_cache = {}  # Cache attention from hooks
        self._hooks = []
        self._guidance_layer_map = {}  # Map DAGA layer to guidance layer
        
        # Freeze backbone initially
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize adaptation modules based on method
        if method == 'daga':
            from core.daga import DAGA
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
            )
            for param in self.daga.parameters():
                param.requires_grad = True
            print(f"✓ Using DAGA on layers {adaptation_layers} (lightweight guidance)")
            
        elif method == 'vit-adapter':
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
        
        elif method == 'full-finetune':
            # Unfreeze ALL backbone parameters
            for param in self.vit.parameters():
                param.requires_grad = True
            total_params = sum(p.numel() for p in self.vit.parameters())
            print(f"✓ Using Full Fine-tuning (all backbone unfrozen)")
            print(f"  Trainable backbone params: {total_params/1e6:.2f}M")
        
        elif method == 'daga-enhanced':
            from core.daga import DAGA
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
            )
            for param in self.daga.parameters():
                param.requires_grad = True
            guidance_layers = [max(0, layer - 1) for layer in adaptation_layers]
            guidance_layers = list(set(guidance_layers))
            self._register_attention_hooks(guidance_layers)
            self._guidance_layer_map = {layer: max(0, layer - 1) for layer in adaptation_layers}
            print(f"✓ Using DAGA-Enhanced on layers {adaptation_layers}")
        
        elif method == 'daga-plus':
            from core.daga import DAGA
            self.daga_plus = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
            for param in self.daga_plus.parameters():
                param.requires_grad = True
            guidance_layers = [max(0, layer - 1) for layer in adaptation_layers]
            guidance_layers = list(set(guidance_layers))
            self._register_attention_hooks(guidance_layers)
            self._guidance_layer_map = {layer: max(0, layer - 1) for layer in adaptation_layers}
            print(f"✓ Using DAGA-Plus on layers {adaptation_layers}")
            
        elif method == 'baseline':
            print("✓ Using Baseline (frozen backbone)")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        print(
            f"✓ ComparisonClassificationModel initialized:\n"
            f"  - Method: {method}\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Adaptation layers: {adaptation_layers}"
        )
    
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
    
    def _get_guidance_map_lightweight(self, x_processed, H, W, num_registers):
        """
        Lightweight forward pass to get guidance layer's attention.
        Only forward until guidance layer, then compute attention and stop.
        ~1.5x faster than full forward pass.
        """
        from core.daga import process_attention_to_guidance
        
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
    
    def _register_attention_hooks(self, layer_indices):
        """Register hooks to capture attention during forward pass (no extra computation)"""
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                x = input[0]
                B, N, C = x.shape
                num_heads = module.num_heads
                head_dim = C // num_heads
                # Capture attention without gradient (minimal overhead)
                with torch.no_grad():
                    qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    q = q * module.scale
                    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                    self._attention_cache[layer_idx] = attn.detach()
            return hook_fn
        
        # Remove existing hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
        # Register new hooks
        for idx in layer_indices:
            if idx < len(self.vit.blocks):
                hook = self.vit.blocks[idx].attn.register_forward_hook(make_hook(idx))
                self._hooks.append(hook)
    
    def _clear_attention_cache(self):
        """Clear cached attention weights"""
        self._attention_cache.clear()
    
    def forward(self, x, request_visualization_maps=False):
        """Forward pass with selected adaptation method"""
        from core.daga import process_attention_to_guidance
        from core.backbones import get_attention_map, compute_daga_guidance_map
        
        B = x.shape[0]
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        adapted_attn_weights = None
        baseline_attn_weights = None
        
        # Compute DAGA guidance using lightweight approach (only forward to guidance layer)
        daga_guidance_map = None
        if self.method in ['daga', 'daga-enhanced', 'daga-plus']:
            daga_guidance_map = self._get_guidance_map_lightweight(
                x_processed, H, W, num_registers
            )
        
        # Clear attention cache at start of forward
        self._clear_attention_cache()
        
        # Forward through blocks with adaptation
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # VPT-Deep: insert prompts before block
            num_prompts_inserted = 0
            if self.method == 'vpt-deep' and idx in self.adaptation_layers:
                x_processed, num_prompts_inserted = self.vpt.insert_prompts(x_processed, idx)
            
            # Extract baseline attention for visualization
            if request_visualization_maps and idx == self.vis_attn_layer:
                with torch.no_grad():
                    baseline_attn_weights = get_attention_map(block, x_processed)
            
            # Pass through block (hooks capture attention automatically for DAGA layers)
            x_processed = block(x_processed, rope_sincos)
            
            # Extract adapted attention for visualization
            if request_visualization_maps and idx == self.vis_attn_layer:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            # VPT-Deep: remove prompts after block
            if self.method == 'vpt-deep' and num_prompts_inserted > 0:
                x_processed = self.vpt.remove_prompts(x_processed, num_prompts_inserted)
            
            # Apply adaptation based on method
            if idx in self.adaptation_layers:
                if self.method in ['daga', 'daga-enhanced'] and daga_guidance_map is not None:
                    # Use pre-computed guidance (from compute_daga_guidance_map)
                    cls_token = x_processed[:, :1, :]
                    register_tokens = x_processed[:, 1:1+num_registers, :]
                    patch_tokens = x_processed[:, 1+num_registers:, :]
                    
                    adapted_patch_tokens = self.daga.apply(
                        patch_tokens, daga_guidance_map, idx, H, W
                    )
                    
                    x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                    
                elif self.method == 'vit-adapter':
                    x_processed = self.adapters.forward(x_processed, idx)
                
                elif self.method == 'adapter-former':
                    x_processed = self.adapters.forward(x_processed, idx)
                
                elif self.method == 'lora':
                    x_processed = self.adapters.forward(x_processed, idx)
                
                elif self.method == 'daga-plus' and daga_guidance_map is not None:
                    cls_token = x_processed[:, :1, :]
                    register_tokens = x_processed[:, 1:1+num_registers, :]
                    patch_tokens = x_processed[:, 1+num_registers:, :]
                    
                    adapted_patch_tokens = self.daga_plus.apply(
                        patch_tokens, daga_guidance_map, idx, H, W
                    )
                    
                    x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                
                # Layer fine-tuning: no additional module, just trainable parameters
        
        x_normalized = self.vit.norm(x_processed)
        features = x_normalized[:, 0]  # CLS token
        logits = self.classifier(features)
        
        return logits, adapted_attn_weights, daga_guidance_map, baseline_attn_weights


def setup_layer_finetuning_optimizer(model, args):
    """
    Setup optimizer for layer-finetuning with lower LR for backbone layers.
    Unfrozen backbone layers use 10x lower LR than classifier.
    """
    from torch.nn.parallel import DataParallel
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    backbone_params = []
    classifier_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                classifier_params.append(param)
            else:
                # Backbone layers (vit.blocks)
                backbone_params.append(param)
    
    # Use much lower LR for layer-finetuning with AdamW
    # AdamW needs lower LR than SGD
    backbone_lr = 1e-5  # Very low LR for pretrained ViT layers
    classifier_lr = 1e-3  # Lower LR for AdamW (not 0.5 which is for SGD)
    
    param_groups = [
        {"params": classifier_params, "lr": classifier_lr, "weight_decay": 0.0},
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": 0.01},
    ]
    
    # Use AdamW for layer-finetuning (more stable than SGD for pretrained weights)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    
    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"✓ Layer-finetuning optimizer:")
    print(f"  Classifier params: {sum(p.numel() for p in classifier_params):,}, LR={classifier_lr}")
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}, LR={backbone_lr}")
    
    return criterion, optimizer, scheduler


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Classification Comparison Experiments")
    
    # Method selection
    parser.add_argument("--method", type=str, required=True, 
                       choices=['baseline', 'full-finetune', 'daga', 'daga-enhanced', 'daga-plus', 'vit-adapter', 'adapter-former', 'lora', 'vpt-deep', 'layer-finetuning'],
                       help="Adaptation method")
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[1, 2, 10, 11],
                       help="Layers to apply adaptation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "sun397", "flowers102", "pets", "cars", "food101", "dtd"], default="cifar100")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Ratio of training data")
    
    # Training arguments
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--output_dir", default="./paper_experiments/outputs")
    parser.add_argument("--enable_swanlab", action="store_true", default=False)
    parser.add_argument("--swanlab_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--vis_indices", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    parser.add_argument("--enable_visualization", action="store_true")
    parser.add_argument("--vis_attn_layer", type=int, default=11)
    
    # Performance optimization
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Enable Mixed Precision (AMP)")
    parser.add_argument("--use_compile", action="store_true", default=False,
                       help="Enable torch.compile() (PyTorch 2.0+)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps. Effective batch = batch_size * steps * num_gpus")
    
    return parser.parse_args()


def main():
    # Setup DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        # Custom experiment name with method
        args.swanlab_name = f"{args.dataset}_{args.method}_L{'-'.join(map(str, args.adaptation_layers))}"
        experiment_name = setup_logging(args, task_name="paper_comparison")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Broadcast output_dir
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Paper Comparison: {args.method.upper()}")
        print(f"DDP Training with {world_size} GPUs")
        print(f"Adaptation layers: {args.adaptation_layers}")
        print(f"Loading {args.dataset.upper()} dataset...")
    
    train_dataset, test_dataset, num_classes = get_classification_dataset(args)
    
    if is_main_process:
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
    
    train_loader, test_loader = create_ddp_dataloaders(
        train_dataset, test_dataset, args.batch_size, world_size, rank,
        num_workers=args.num_workers
    )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # Create model with selected method (all methods use ComparisonClassificationModel)
    model = ComparisonClassificationModel(
            vit_model,
            num_classes=num_classes,
            method=args.method,
            adaptation_layers=args.adaptation_layers,
            enable_visualization=args.enable_visualization,
            vis_attn_layer=args.vis_attn_layer,
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
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    
    if is_main_process:
        print(f"✓ Model wrapped with DDP on {world_size} GPUs ({args.method})")
        print(f"✓ Optimizations: AMP={args.use_amp}, Compile={args.use_compile}\n")
    
    # Prepare visualization data
    fixed_vis_images = None
    if is_main_process and args.enable_visualization:
        print("📸 Preparing visualization data...")
        fixed_vis_images = prepare_visualization_data(test_dataset, args, device)
    
    # Setup training components
    # For layer-finetuning, use lower LR for unfrozen backbone layers
    if args.method == 'layer-finetuning':
        criterion, optimizer, scheduler = setup_layer_finetuning_optimizer(model, args)
    else:
        criterion, optimizer, scheduler = setup_training_components(model, args)
    
    # Run training
    if is_main_process:
        effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
        print(f"\n{'='*70}")
        print(f"Starting training with {args.method}")
        if args.gradient_accumulation_steps > 1:
            print(f"⚡ Gradient Accumulation: {args.gradient_accumulation_steps} steps")
            print(f"   Effective batch size: {args.batch_size} × {args.gradient_accumulation_steps} × {world_size} GPUs = {effective_batch}")
        print(f"{'='*70}\n")
    
    best_acc, final_acc, total_time = run_training_loop(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        device, args, output_dir, fixed_vis_images, test_dataset,
        rank=rank, world_size=world_size, use_amp=args.use_amp,
        accumulation_steps=args.gradient_accumulation_steps
    )
    
    if is_main_process:
        finalize_experiment(
            best_acc, final_acc, total_time, output_dir,
            enable_swanlab=getattr(args, 'enable_swanlab', True)
        )
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

