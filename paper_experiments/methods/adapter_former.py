"""
AdapterFormer: Adapter for Vision Transformers
Based on: "AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition" (NeurIPS 2022)

Key difference from ViT-Adapter:
- AdapterFormer uses parallel adapter branches instead of sequential
- Inserted parallel to the FFN in each transformer block
"""
import torch
import torch.nn as nn


class AdapterFormerModule(nn.Module):
    """
    Single AdapterFormer module with parallel structure.
    Adds a parallel branch to the FFN in transformer blocks.
    """
    def __init__(self, embed_dim=768, bottleneck_dim=64, dropout=0.0, init_scale=1e-4):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        # Non-linearity
        self.act = nn.GELU()
        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable scale factor (init small for stable training)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) input features
        Returns:
            (B, N, D) adapter output (to be added to main branch)
        """
        down = self.down_proj(x)
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        return self.scale * up


class AdapterFormer(nn.Module):
    """
    AdapterFormer: Adapters for Vision Transformers
    
    Creates parallel adapter branches for specified transformer layers.
    Each adapter is applied in parallel to the FFN output.
    """
    def __init__(self, embed_dim=768, adapter_layers=None, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        self.adapter_layers = adapter_layers if adapter_layers else []
        self.embed_dim = embed_dim
        
        # Create adapter for each specified layer
        self.adapters = nn.ModuleDict({
            str(i): AdapterFormerModule(
                embed_dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout
            ) for i in self.adapter_layers
        })
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"✓ AdapterFormer initialized")
        print(f"  Layers: {self.adapter_layers}")
        print(f"  Bottleneck dim: {bottleneck_dim}")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    
    def forward(self, x, layer_idx):
        """
        Apply adapter to features at specified layer.
        Returns additive residual to be added to main branch.
        
        Args:
            x: (B, N, D) features from transformer block output
            layer_idx: current layer index
        Returns:
            (B, N, D) features with adapter applied
        """
        if layer_idx in self.adapter_layers and str(layer_idx) in self.adapters:
            adapter_out = self.adapters[str(layer_idx)](x)
            return x + adapter_out
        return x
    
    def get_adapter_scale(self, layer_idx):
        """Get the learned scale factor for a specific layer"""
        if str(layer_idx) in self.adapters:
            return self.adapters[str(layer_idx)].scale.item()
        return None


class AdapterFormerConfig:
    """Configuration for AdapterFormer experiments"""
    def __init__(self, adapter_layers, bottleneck_dim=64, dropout=0.0):
        self.adapter_layers = adapter_layers
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
    
    def __repr__(self):
        return (f"AdapterFormerConfig(layers={self.adapter_layers}, "
                f"bottleneck={self.bottleneck_dim}, dropout={self.dropout})")

