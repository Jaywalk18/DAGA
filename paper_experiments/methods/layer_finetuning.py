"""
Layer-wise Fine-tuning: Unfreeze and fine-tune specific transformer layers
Comparison baseline for DAGA paper
"""
import torch
import torch.nn as nn


def unfreeze_layers(vit_model, layers_to_unfreeze):
    """
    Unfreeze specific layers in the ViT model for fine-tuning
    
    Args:
        vit_model: DINOv3 ViT model
        layers_to_unfreeze: List of layer indices to unfreeze (e.g., [1, 2, 10, 11])
    
    Returns:
        Number of trainable parameters
    """
    # First, freeze everything
    for param in vit_model.parameters():
        param.requires_grad = False
    
    # Unfreeze specified layers
    trainable_params = 0
    for idx in layers_to_unfreeze:
        if idx < len(vit_model.blocks):
            block = vit_model.blocks[idx]
            for param in block.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
    
    print(f"✓ Layer-wise Fine-tuning initialized")
    print(f"  Unfrozen layers: {layers_to_unfreeze}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return trainable_params


def get_layer_lr_groups(vit_model, base_lr, layers_to_unfreeze):
    """
    Create parameter groups with layer-wise learning rates
    Often lower layers need smaller LR
    
    Args:
        vit_model: DINOv3 model
        base_lr: Base learning rate
        layers_to_unfreeze: List of layer indices
    
    Returns:
        List of parameter groups for optimizer
    """
    param_groups = []
    
    for idx in layers_to_unfreeze:
        if idx < len(vit_model.blocks):
            # Use smaller LR for earlier layers
            layer_lr = base_lr * (0.1 if idx < 6 else 1.0)
            
            param_groups.append({
                'params': vit_model.blocks[idx].parameters(),
                'lr': layer_lr,
                'name': f'layer_{idx}'
            })
    
    return param_groups


class LayerFineTuningConfig:
    """Configuration for layer-wise fine-tuning experiments"""
    def __init__(self, layers_to_unfreeze, base_lr=2e-2, use_layer_wise_lr=False):
        self.layers_to_unfreeze = layers_to_unfreeze
        self.base_lr = base_lr
        self.use_layer_wise_lr = use_layer_wise_lr
    
    def __repr__(self):
        return (f"LayerFineTuningConfig(layers={self.layers_to_unfreeze}, "
                f"base_lr={self.base_lr}, layer_wise_lr={self.use_layer_wise_lr})")

