"""
DAGA: Dynamic Attention-Guided Adaptation

Core innovation:
- Attention-as-Guidance: frozen backbone attention → dynamic gate
- Instance-specific adaptation: each sample gets unique gate

Architecture:
- Shared feature extractor across layers (reduce redundant conv)
- Learnable spatial scale (model decides if spatial is needed)
- Feature caching for efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class SharedFeatureExtractor(nn.Module):
    """Shared conv layers for all DAGA layers, run once per forward."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, attention_map: torch.Tensor) -> torch.Tensor:
        x = attention_map.unsqueeze(1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class LayerSpecificHead(nn.Module):
    """Layer-specific guidance and spatial weight extraction."""
    
    def __init__(self, hidden_dim: int = 64, guidance_dim: int = 128):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.guidance_fc = nn.Sequential(
            nn.Linear(hidden_dim, guidance_dim),
            nn.GELU(),
            nn.Linear(guidance_dim, guidance_dim),
            nn.LayerNorm(guidance_dim),
        )
        self.spatial_fc = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, shared_features: torch.Tensor, H: int, W: int) -> tuple:
        global_feat = self.global_pool(shared_features).flatten(1)
        guidance = self.guidance_fc(global_feat)
        
        spatial_weight = self.spatial_fc(shared_features)
        spatial_weight = F.interpolate(spatial_weight, size=(H, W), mode='bilinear', align_corners=False)
        spatial_weight = torch.sigmoid(spatial_weight).squeeze(1)
        
        return guidance, spatial_weight


class AdaptationLayer(nn.Module):
    """
    Dynamic gate layer with learnable spatial importance.
    
    Key: spatial_scale is learnable and initialized small.
    - Classification tasks: model learns spatial_scale → 0
    - Detection tasks: model learns spatial_scale → higher value
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        guidance_dim: int = 128,
        mlp_ratio: float = 0.25,
        drop_rate: float = 0.0,
        layer_idx: int = 0,
        total_layers: int = 4,
    ):
        super().__init__()
        bottleneck_dim = int(feature_dim * mlp_ratio)
        
        # Dynamic channel gate from attention guidance
        self.channel_gate = nn.Sequential(
            nn.Linear(guidance_dim, feature_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_gate[0].weight)
        nn.init.zeros_(self.channel_gate[0].bias)
        
        # Bottleneck transformation
        self.down_proj = nn.Linear(feature_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, feature_dim)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
        # Learnable scales
        layer_ratio = layer_idx / max(total_layers - 1, 1)
        self.scale = nn.Parameter(torch.ones(1) * (0.1 + 0.05 * layer_ratio))
        
        # Spatial scale: initialized small, let model learn if needed
        # Classification will learn to ignore spatial (→0)
        # Detection will learn to use spatial (→higher)
        self.spatial_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        patch_features: torch.Tensor,
        guidance: torch.Tensor,
        spatial_weight: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = patch_features.shape
        
        # Channel gate from guidance (core DAGA innovation)
        channel_gate = self.channel_gate(guidance).unsqueeze(1)
        
        # Spatial modulation (learnable importance)
        spatial_mod = spatial_weight.reshape(B, -1, 1)
        if spatial_mod.shape[1] != N:
            H = W = int(N ** 0.5)
            spatial_mod = F.adaptive_avg_pool2d(
                spatial_weight.unsqueeze(1), (H, W)
            ).reshape(B, -1, 1)
        
        # Combined gate: channel * (1 + spatial_scale * spatial)
        # If spatial_scale→0, degrades to pure channel gate
        combined_gate = channel_gate * (1.0 + self.spatial_scale * spatial_mod)
        
        # Bottleneck
        delta = self.down_proj(patch_features)
        delta = self.act(delta)
        delta = self.drop(delta)
        delta = self.up_proj(delta)
        
        return patch_features + self.scale * (combined_gate * delta)


class DAGA(nn.Module):
    """
    DAGA: Dynamic Attention-Guided Adaptation
    
    Core preserved:
    - Dynamic gate from attention (instance-specific)
    - Attention as guidance signal
    
    Features:
    - Shared conv feature extraction (4x→1x)
    - Learnable spatial importance (no manual task_mode)
    - Feature caching
    
    Usage:
    - Multi-layer mode: daga = DAGA(daga_layers=[1,2,10,11]); daga.apply(features, attn, layer_idx, H, W)
    - Single-layer mode: daga = DAGA(daga_layers=[11]); daga(features, attention_map)  # calls forward()
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        daga_layers: List[int] = [1, 2, 10, 11],
        guidance_dim: int = 128,
        mlp_ratio: float = 0.25,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.daga_layers = daga_layers
        self.feature_dim = feature_dim
        
        # Shared feature extractor (optimization)
        self.shared_extractor = SharedFeatureExtractor(hidden_dim=64)
        
        # Layer-specific heads
        self.layer_heads = nn.ModuleDict({
            str(idx): LayerSpecificHead(hidden_dim=64, guidance_dim=guidance_dim)
            for idx in daga_layers
        })
        
        # Adaptation layers with learnable spatial scale
        self.layers = nn.ModuleDict({
            str(idx): AdaptationLayer(
                feature_dim=feature_dim,
                guidance_dim=guidance_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                layer_idx=i,
                total_layers=len(daga_layers),
            ) for i, idx in enumerate(daga_layers)
        })
        
        # Learnable mix weight for controlling DAGA strength (for compatibility with some scripts)
        self.mix_weight = nn.Parameter(torch.ones(1) * 1.0)
        
        # Cache for shared features
        self._cached_shared_features: Optional[torch.Tensor] = None
        self._cached_attention_shape: Optional[tuple] = None
    
    def forward(self, patch_features: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single-layer DAGA mode (used by main_dinotxt.py).
        
        Args:
            patch_features: (B, N, D) patch token features
            attention_map: (B, H, W) attention guidance map
            
        Returns:
            adapted_features: (B, N, D) adapted patch features
        """
        B, H, W = attention_map.shape
        
        # Use the first (or only) layer in daga_layers
        layer_idx = self.daga_layers[0]
        
        # Apply DAGA through the apply method
        adapted = self.apply(patch_features, attention_map, layer_idx, H, W)
        
        # Apply mix_weight for smooth control
        return patch_features + self.mix_weight * (adapted - patch_features)
    
    def encode_attention(self, attention_map: torch.Tensor, layer_idx: int) -> tuple:
        """Encode attention map to guidance + spatial weight."""
        if str(layer_idx) not in self.layer_heads:
            return None, None
        
        B, H, W = attention_map.shape
        
        # Only use cache during inference (training needs fresh graph for each backward)
        if not self.training:
            if (self._cached_shared_features is not None and 
                self._cached_attention_shape == (B, H, W)):
                shared_features = self._cached_shared_features
            else:
                shared_features = self.shared_extractor(attention_map)
                self._cached_shared_features = shared_features
                self._cached_attention_shape = (B, H, W)
        else:
            # Training: always compute fresh (no cache to avoid backward issues)
            shared_features = self.shared_extractor(attention_map)
        
        return self.layer_heads[str(layer_idx)](shared_features, H, W)
    
    def apply(
        self,
        patch_features: torch.Tensor,
        attention_map: torch.Tensor,
        layer_idx: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Apply DAGA adaptation at specified layer."""
        if str(layer_idx) not in self.layers:
            return patch_features
        
        guidance, spatial_weight = self.encode_attention(attention_map, layer_idx)
        return self.layers[str(layer_idx)](patch_features, guidance, spatial_weight)
    
    def clear_cache(self):
        self._cached_shared_features = None
        self._cached_attention_shape = None
    
    def get_param_count(self) -> Dict[str, int]:
        shared = sum(p.numel() for p in self.shared_extractor.parameters())
        heads = sum(sum(p.numel() for p in h.parameters()) for h in self.layer_heads.values())
        layers = sum(p.numel() for p in self.layers.parameters())
        return {'shared': shared, 'heads': heads, 'layers': layers, 'total': shared + heads + layers}
    
    def get_spatial_scales(self) -> Dict[int, float]:
        """Get learned spatial scales for analysis."""
        return {int(k): self.layers[k].spatial_scale.item() for k in self.layers.keys()}


def create_daga(
    feature_dim: int = 768,
    daga_layers: List[int] = [1, 2, 10, 11],
    drop_rate: float = 0.1,
    mlp_ratio: float = 0.25,
) -> DAGA:
    """Create a DAGA module with specified configuration."""
    return DAGA(
        feature_dim=feature_dim,
        daga_layers=daga_layers,
        guidance_dim=128,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
    )


# =============================================================================
# Utility functions for attention processing
# =============================================================================

def process_attention_to_guidance(attn_weights, H, W, num_registers):
    """Convert attention weights to guidance map (B, H, W)"""
    num_patches = H * W
    # CLS token attention to all other tokens
    cls_attn_all = attn_weights[:, :, 0, 1:]  # (B, num_heads, N-1)
    # Skip register tokens
    patch_start = num_registers
    cls_attn_patches = cls_attn_all[:, :, patch_start:]
    # Average over heads
    cls_attn = cls_attn_patches.mean(dim=1)  # (B, num_patches)
    
    # Normalize to [0, 1]
    min_val = cls_attn.amin(dim=1, keepdim=True)
    max_val = cls_attn.amax(dim=1, keepdim=True)
    cls_attn = (cls_attn - min_val) / (max_val - min_val + 1e-8)
    
    if cls_attn.shape[1] == num_patches:
        return cls_attn.reshape(-1, H, W)
    return None


class AttentionHook:
    """Simple hook to capture attention weights during forward pass"""
    
    def __init__(self):
        self.hook = None
        self.attention = None
    
    def register(self, block):
        """Register hook on a transformer block's attention module"""
        if self.hook is not None:
            self.hook.remove()
        self.hook = block.attn.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        x = input[0]
        B, N, C = x.shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        
        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            self.attention = attn.softmax(dim=-1).detach()
    
    def get(self):
        return self.attention
    
    def clear(self):
        self.attention = None
    
    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


class AttentionHookManager:
    """Manager for multiple attention hooks across transformer layers"""
    
    def __init__(self):
        self.hooks = {}
    
    def register_hooks(self, vit, layer_indices):
        """Register attention hooks on specified transformer blocks"""
        self.remove_hooks()  # Clean up any existing hooks
        for idx in layer_indices:
            if idx < len(vit.blocks):
                hook = AttentionHook()
                hook.register(vit.blocks[idx])
                self.hooks[idx] = hook
    
    def get_attention(self, layer_idx):
        """Get captured attention from a specific layer"""
        if layer_idx in self.hooks:
            return self.hooks[layer_idx].get()
        return None
    
    def clear_all(self):
        """Clear all captured attention weights"""
        for hook in self.hooks.values():
            hook.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


# Backward compatibility aliases
DAGAPlusUnified = DAGA
create_daga_plus_unified = create_daga


if __name__ == '__main__':
    import time
    
    feature_dim = 768
    daga_layers = [1, 2, 10, 11]
    B, H, W, N, D = 2, 37, 37, 37*37, feature_dim
    
    print("=" * 50)
    print("DAGA Test")
    print("=" * 50)
    
    daga = create_daga(feature_dim, daga_layers)
    params = daga.get_param_count()
    print(f"Params: {params['total']:,}")
    print(f"  shared: {params['shared']:,}, heads: {params['heads']:,}, layers: {params['layers']:,}")
    
    attention_map = torch.rand(B, H, W)
    patch_features = torch.rand(B, N, D)
    
    for layer_idx in daga_layers:
        adapted = daga.apply(patch_features, attention_map, layer_idx, H, W)
        print(f"Layer {layer_idx}: {patch_features.shape} → {adapted.shape}")
    
    print(f"\nInitial spatial_scales: {daga.get_spatial_scales()}")
    
    # Speed test
    start = time.time()
    for _ in range(50):
        daga.clear_cache()
        for layer_idx in daga_layers:
            _ = daga.apply(patch_features, attention_map, layer_idx, H, W)
    print(f"Speed (CPU): {50*4/(time.time()-start):.1f} layer-apps/sec")
    
    print("\n✓ Test passed!")
