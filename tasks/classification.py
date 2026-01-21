import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import swanlab

from core.daga import DAGA, process_attention_to_guidance
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class ClassificationModel(nn.Module):
    """
    Classification model with optimized DAGA inference.
    
    Optimization strategy (Two-Stage with Last-Layer Guidance):
    - Stage 1: Lightweight forward (no_grad) to get last layer's attention
    - Stage 2: Normal forward with DAGA adaptation using last layer's attention
    
    Benefits:
    - Uses last layer's attention for best accuracy (like original design)
    - Still faster than original because Stage 1 has no gradient computation
    - ~1.5x speedup compared to original (Stage 1 is very lightweight)
    """
    def __init__(
        self,
        pretrained_vit,
        num_classes=10,
        use_daga=False,
        daga_layers=[11],
        enable_visualization=False,
        vis_attn_layer=11,
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.feature_dim = self.vit.embed_dim
        self.enable_visualization = enable_visualization
        self.vis_attn_layer = vis_attn_layer
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        self.captured_attn = None
        self.captured_guidance_attn = None
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if self.use_daga:
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=daga_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
        
        # Use simple linear layer like raw_code (not ClassificationHead)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Ensure classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        if self.use_daga:
            for param in self.daga.parameters():
                param.requires_grad = True
        
        print(
            f"✓ ClassificationModel initialized (Optimized Two-Stage):\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})\n"
            f"  - Guidance layer: {self.daga_guidance_layer_idx}"
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
        Lightweight forward pass to get last layer's attention (no gradients).
        Faster than full forward because no gradient computation or activation storage.
        Only forward until guidance layer, then compute attention and stop.
        """
        with torch.no_grad():
            features = x_processed.clone()
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                
                # Capture attention at guidance layer
                if idx == self.daga_guidance_layer_idx:
                    attn_weights = self._get_attention_inline(block, features)
                    guidance_map = process_attention_to_guidance(
                        attn_weights, H, W, num_registers
                    )
                    return guidance_map
                
                features = block(features, rope_sincos)
        
        return None
    
    def forward(self, x, request_visualization_maps=False):
        """
        Optimized forward pass with Two-Stage strategy.
        
        Stage 1: Lightweight forward (no_grad) to get last layer's attention
        Stage 2: Normal forward with DAGA adaptation
        
        ~1.5x faster than original compute_daga_guidance_map() approach.
        """
        B = x.shape[0]
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        # Stage 1: Get guidance map from last layer (lightweight, no gradients)
        daga_guidance_map = None
        if self.use_daga:
            daga_guidance_map = self._get_guidance_map_lightweight(
                x_processed, H, W, num_registers
            )
        
        baseline_attn_weights = None  # Attention before DAGA influence
        adapted_attn_weights = None   # Attention AFTER DAGA modification
        
        # Calculate which layer to capture adapted attention
        # Use the last DAGA layer itself (not layer+1) to show DAGA's effect on attention
        last_daga_layer = max(self.daga_layers) if self.use_daga and self.daga_layers else -1
        # Capture adapted attention at the last DAGA layer (after DAGA is applied)
        adapted_attn_layer = last_daga_layer if last_daga_layer >= 0 else -1
        
        # Stage 2: Forward with DAGA adaptation using pre-computed guidance
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Extract BASELINE attention at vis_attn_layer (before DAGA influence)
            if request_visualization_maps and idx == self.vis_attn_layer:
                with torch.no_grad():
                    baseline_attn_weights = get_attention_map(block, x_processed)
            
            # Pass tokens through the block
            x_processed = block(x_processed, rope_sincos)
            
            # Apply DAGA AFTER block forward (modifies block output)
            if (
                self.use_daga
                and idx in self.daga_layers
                and daga_guidance_map is not None
            ):
                cls_token = x_processed[:, :1, :]
                register_start_index = 1
                register_end_index = 1 + num_registers
                register_tokens = x_processed[:, register_start_index:register_end_index, :]
                patch_start_index = 1 + num_registers
                patch_tokens = x_processed[:, patch_start_index:, :]
                
                # Store original for comparison
                patch_tokens_before = patch_tokens.clone() if request_visualization_maps and idx == self.vis_attn_layer else None
                
                adapted_patch_tokens = self.daga.apply(
                    patch_tokens, daga_guidance_map, idx, H, W
                )
                
                # Debug: Print feature change magnitude during visualization
                if request_visualization_maps and idx == self.vis_attn_layer:
                    feature_diff = (adapted_patch_tokens - patch_tokens_before).abs().mean().item()
                    print(f"  [Layer {idx}] DAGA feature change: {feature_diff:.6f}")
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            # Extract ADAPTED attention at the last DAGA layer (AFTER DAGA is applied)
            # This shows how DAGA modifies the attention distribution
            if request_visualization_maps and self.use_daga and idx == adapted_attn_layer:
                with torch.no_grad():
                    # Re-compute attention with DAGA-modified features
                    adapted_attn_weights = get_attention_map(block, x_processed)
        
        x_normalized = self.vit.norm(x_processed)
        # Use only CLS token for classification (matching raw_code)
        features = x_normalized[:, 0]  # (B, C)
        logits = self.classifier(features)
        
        # For visualization: return adapted attention if available, otherwise baseline
        # Also return baseline as guidance_map for comparison
        final_attn = adapted_attn_weights if adapted_attn_weights is not None else baseline_attn_weights
        return logits, final_attn, daga_guidance_map, baseline_attn_weights


def setup_training_components(model, args):
    """Setup criterion, optimizer, scheduler
    
    Args:
        model: The model to train
        args: Training arguments
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Handle both DataParallel and DistributedDataParallel
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    daga_params = []
    classifier_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                classifier_params.append(param)
    
    # Use the learning rate directly as specified (no scaling)
    # DDP already handles gradient averaging across GPUs
    lr = args.lr
    
    param_groups = [{"params": classifier_params, "lr": lr, "weight_decay": 0.0}]
    if daga_params:
        param_groups.append(
            {"params": daga_params, "lr": lr * 0.5, "weight_decay": 0.0}
        )
    
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    
    warmup_epochs = 1
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Avoid division by zero for single epoch training
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, rank=0, 
                scaler=None, use_amp=False, accumulation_steps=1):
    """Train for one epoch with optional AMP support, DDP sync, and gradient accumulation
    
    Args:
        accumulation_steps: Number of steps to accumulate gradients before optimizer step.
                           Effective batch size = batch_size * accumulation_steps * num_gpus
    """
    import torch.distributed as dist
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    is_main = (rank == 0)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", disable=not is_main)
    
    optimizer.zero_grad(set_to_none=True)  # Zero grad at start
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _, _, _ = model(images, request_visualization_maps=False)
            loss = criterion(logits, labels)
            # Scale loss by accumulation steps for correct gradient magnitude
            loss = loss / accumulation_steps
        
        # Backward with scaler if using AMP
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps  # Rescale for logging
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        running_avg_loss = total_loss / (batch_idx + 1)
        
        if is_main:
            pbar.set_postfix(
                {
                    "Loss": f"{running_avg_loss:.4f}",
                    "Acc": f"{100. * correct / total:.2f}%",
                }
            )
    
    # Sync metrics across all GPUs in DDP
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([total_loss], dtype=torch.float64, device=device)
        correct_tensor = torch.tensor([correct], dtype=torch.float64, device=device)
        total_tensor = torch.tensor([total], dtype=torch.float64, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        total_loss = loss_tensor.item() / world_size  # Average loss
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device, rank=0, use_amp=False):
    """Evaluate model with optional AMP support and DDP sync"""
    import torch.distributed as dist
    
    model.eval()
    correct, total = 0, 0
    is_main = (rank == 0)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", disable=not is_main):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _, _, _ = model(images, request_visualization_maps=False)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Sync across all GPUs in DDP
    if dist.is_available() and dist.is_initialized():
        correct_tensor = torch.tensor([correct], dtype=torch.float64, device=device)
        total_tensor = torch.tensor([total], dtype=torch.float64, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    return 100.0 * correct / total if total > 0 else 0.0


def visualize_attention_comparison(
    model, fixed_images, args, output_dir, epoch, test_dataset=None
):
    """Generate attention comparison visualizations
    
    Baseline: 1x2 layout - [Original Image, Frozen Backbone Attn]
    DAGA: 1x4 layout - [Original Image, Frozen Backbone Attn, Adapted Attn, Adapted Attn Overlay]
    """
    from scipy.ndimage import zoom
    
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    class_names = getattr(test_dataset, "classes", None)
    
    # Get actual model from DDP wrapper if needed
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    actual_base_model = base_model.module if isinstance(base_model, (DataParallel, DDP)) else base_model
    
    is_daga_model = getattr(actual_base_model, "use_daga", False)
    
    # Print DAGA status
    if is_daga_model:
        print(f"\n[DAGA] Status at epoch {epoch+1}:")
        print(f"  Layers: {actual_base_model.daga_layers}")
        print(f"  Spatial scales: {actual_base_model.daga.get_spatial_scales()}")
    
    with torch.no_grad():
        _, (H, W) = actual_base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        vis_attn_layer = actual_base_model.vis_attn_layer
        
        # Always get baseline attention (frozen backbone)
        x_proc, _ = actual_base_model.vit.prepare_tokens_with_masks(fixed_images)
        baseline_raw_weights = None
        for i in range(vis_attn_layer + 1):
            rope_sincos = (
                actual_base_model.vit.rope_embed(H=H, W=W)
                if actual_base_model.vit.rope_embed
                else None
            )
            if i == vis_attn_layer:
                baseline_raw_weights = get_attention_map(
                    actual_base_model.vit.blocks[i], x_proc
                )
            x_proc = actual_base_model.vit.blocks[i](x_proc, rope_sincos)
        
        if baseline_raw_weights is None:
            print(f"⚠ Warning: Failed to extract baseline attention from layer {vis_attn_layer}.")
            return []
        
        baseline_attn_np = process_attention_weights(baseline_raw_weights, num_patches_expected, H, W)
        if baseline_attn_np is None:
            return []
        
        # Get adapted attention and predictions (only for DAGA model)
        adapted_attn_np = None
        daga_guidance_map = None
        
        logits, adapted_attn_weights, daga_guidance_map, baseline_attn_weights = base_model(
            fixed_images, request_visualization_maps=True
        )
        predictions = logits.argmax(dim=1).cpu().numpy()
        
        if is_daga_model and adapted_attn_weights is not None:
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches_expected, H, W)
        
        # Prepare images
        images_np = fixed_images.cpu().numpy()
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            original_image_index = args.vis_indices[j]
            actual_label_name = "Unknown"
            pred_label_name = "Unknown"
            
            if class_names and hasattr(test_dataset, "targets"):
                try:
                    actual_label_int = test_dataset.targets[original_image_index]
                    actual_label_name = class_names[actual_label_int]
                    pred_label_name = class_names[predictions[j]]
                except (IndexError, TypeError):
                    actual_label_name = "Label Index Error"
            
            pred_correct = "✓" if actual_label_name == pred_label_name else "✗"
            fig_title = f"Epoch {epoch+1} - Img#{original_image_index} | True: {actual_label_name} | Pred: {pred_label_name} {pred_correct}"
            
            # Denormalize image
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            if is_daga_model and adapted_attn_np is not None:
                # DAGA model: 1x4 layout
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                fig.suptitle(fig_title, fontsize=14, fontweight="bold")
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                im1 = axes[1].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[1].set_title(f"Frozen Backbone Attn (L{vis_attn_layer})")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                im2 = axes[2].imshow(adapted_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[2].set_title(f"Adapted Attn (After DAGA, L{vis_attn_layer})")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                axes[3].imshow(img)
                attn_map = adapted_attn_np[j]
                if attn_map.shape != img.shape[:2]:
                    zoom_h = img.shape[0] / attn_map.shape[0]
                    zoom_w = img.shape[1] / attn_map.shape[1]
                    attn_resized = zoom(attn_map, (zoom_h, zoom_w), order=1)
                else:
                    attn_resized = attn_map
                im3 = axes[3].imshow(attn_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                axes[3].set_title("Adapted Attn Overlay")
                axes[3].axis("off")
                plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            else:
                # Baseline: 1x2 layout
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(fig_title, fontsize=14, fontweight="bold")
                
                # Panel 1: Original Image
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                # Panel 2: Frozen Backbone Attention
                im1 = axes[1].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[1].set_title(f"Frozen Backbone Attn (L{vis_attn_layer})")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            vis_figs.append(fig)
            
            fig.savefig(
                vis_save_path / f"epoch_{epoch+1}_imgidx_{original_image_index}.png",
                dpi=100,
            )
            plt.close(fig)
    
    return vis_figs


def prepare_visualization_data(test_dataset, args, device):
    """Prepare fixed batch of images for visualization"""
    if not args.enable_visualization:
        return None
    
    max_idx = len(test_dataset) - 1
    valid_indices = [idx for idx in args.vis_indices if idx <= max_idx]
    
    if len(valid_indices) < len(args.vis_indices):
        print(f"Warning: Some vis_indices exceed test set size ({len(test_dataset)})")
        print(f"   Original indices: {args.vis_indices}")
        if len(valid_indices) == 0:
            import random
            valid_indices = random.sample(range(len(test_dataset)), min(4, len(test_dataset)))
        args.vis_indices = valid_indices
        print(f"   Adjusted indices: {args.vis_indices}")
    
    print(f"\n📸 Preparing visualization data for image indices: {args.vis_indices}...")
    vis_subset = torch.utils.data.Subset(test_dataset, args.vis_indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(args.vis_indices), shuffle=False)
    fixed_images, _ = next(iter(vis_loader))
    print("✓ Visualization images loaded.")
    return fixed_images.to(device)


def run_training_loop(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    output_dir,
    fixed_vis_images,
    test_dataset=None,
    rank=0,
    world_size=1,
    use_amp=False,
    accumulation_steps=1,
):
    """Execute main training and evaluation loop with optional AMP support
    
    Args:
        accumulation_steps: Gradient accumulation steps. Effective batch = batch_size * accumulation_steps * num_gpus
    """
    from core.utils import TopKCheckpointManager, save_results_json
    
    is_main_process = (rank == 0)
    best_acc = 0.0
    start_time = time.time()
    
    # Initialize checkpoint manager (keep top 3)
    ckpt_manager = TopKCheckpointManager(output_dir, k=3, metric_higher_is_better=True) if is_main_process else None
    
    # Track epoch history for JSON
    epoch_history = []
    
    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    
    if is_main_process and use_amp:
        print("⚡ Mixed Precision (AMP) enabled")
    
    # Helper to save results JSON (called after each epoch)
    def update_results_json():
        method = getattr(args, 'method', 'daga' if getattr(args, 'use_daga', False) else 'baseline')
        results = {
            "method": method,
            "dataset": args.dataset,
            "best_acc": best_acc,
            "current_epoch": len(epoch_history),
            "total_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "total_time_minutes": (time.time() - start_time) / 60,
            "epoch_history": epoch_history,
        }
        if hasattr(args, 'adaptation_layers'):
            results["adaptation_layers"] = args.adaptation_layers
        if hasattr(args, 'daga_layers'):
            results["daga_layers"] = args.daga_layers
        save_results_json(output_dir, results)
    
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            rank=rank, scaler=scaler, use_amp=use_amp,
            accumulation_steps=accumulation_steps
        )
        test_acc = evaluate(model, test_loader, device, rank=rank, use_amp=use_amp)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        # Only print on main process
        if is_main_process:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
            print(
                f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
            )
            print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
            
            # Track history
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })
        
        if is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_time_minutes": elapsed_time / 60,
            }
            
            if args.enable_visualization and fixed_vis_images is not None and (
                epoch % args.log_freq == 0 or epoch == args.epochs - 1
            ):
                print("📊 Generating attention visualizations...")
                vis_figs = visualize_attention_comparison(
                    model, fixed_vis_images, args, output_dir, epoch, test_dataset
                )
                if vis_figs:
                    log_dict["attention_comparison"] = [
                        swanlab.Image(fig) for fig in vis_figs
                    ]
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
        
        # Update top-K checkpoints
        if is_main_process and ckpt_manager is not None:
            is_best, current_best = ckpt_manager.update(model, optimizer, epoch, test_acc, args)
            if is_best and current_best > best_acc:
                best_acc = current_best
                print(f"   ✅ New best! (Test Acc: {best_acc:.2f}%)")
        
        # Update results JSON after each epoch
        if is_main_process:
            update_results_json()
    
    # Save final model
    if is_main_process:
        ckpt_manager.save_final(model, optimizer, args.epochs - 1, test_acc, args)
        # Final update with final_acc
        epoch_history[-1]["final"] = True
        update_results_json()
        print(f"   📄 Results saved to {output_dir}/results.json")
    
    return best_acc, test_acc, (time.time() - start_time) / 60
