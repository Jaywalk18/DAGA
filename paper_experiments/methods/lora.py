"""
LoRA: Low-Rank Adaptation for Vision Transformers
Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
"""
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.
    Adds low-rank matrices A and B to the original weight: W' = W + BA
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros (so initial output is zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, original_weight, original_bias=None):
        """
        Args:
            x: input tensor
            original_weight: frozen original weight matrix
            original_bias: frozen original bias (optional)
        """
        # Original output
        out = nn.functional.linear(x, original_weight, original_bias)
        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out + lora_out


class LoRAAttention(nn.Module):
    """
    LoRA adapter for transformer attention.
    Applies LoRA to Q, K, V projections.
    """
    def __init__(self, embed_dim, num_heads, rank=4, alpha=1.0, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # LoRA for Q, K, V
        self.lora_q = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha)
        self.lora_k = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha)
        self.lora_v = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  LoRA params per layer: {total_params:,}")
    
    def forward(self, x, original_qkv_weight, original_qkv_bias=None):
        """
        Apply LoRA to QKV projection.
        
        Args:
            x: (B, N, D) input features
            original_qkv_weight: (3*D, D) frozen QKV weight
            original_qkv_bias: (3*D,) frozen QKV bias
        """
        B, N, D = x.shape
        
        # Split original weights
        q_weight = original_qkv_weight[:D]
        k_weight = original_qkv_weight[D:2*D]
        v_weight = original_qkv_weight[2*D:]
        
        if original_qkv_bias is not None:
            q_bias = original_qkv_bias[:D]
            k_bias = original_qkv_bias[D:2*D]
            v_bias = original_qkv_bias[2*D:]
        else:
            q_bias = k_bias = v_bias = None
        
        # Apply LoRA
        q = self.lora_q(x, q_weight, q_bias)
        k = self.lora_k(x, k_weight, k_bias)
        v = self.lora_v(x, v_weight, v_bias)
        
        return q, k, v


class LoRA(nn.Module):
    """
    LoRA module for Vision Transformers.
    Applies low-rank adaptation to specified transformer layers.
    """
    def __init__(self, embed_dim=768, lora_layers=None, rank=4, alpha=1.0):
        super().__init__()
        self.lora_layers = lora_layers if lora_layers else []
        self.embed_dim = embed_dim
        self.rank = rank
        self.alpha = alpha
        
        # Create LoRA adapters for each specified layer
        # Simplified version: apply to the output of each block
        self.lora_modules = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(embed_dim, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, embed_dim, bias=False),
            ) for i in self.lora_layers
        })
        
        # Initialize with small values
        for module in self.lora_modules.values():
            nn.init.kaiming_uniform_(module[0].weight, a=math.sqrt(5))
            nn.init.zeros_(module[2].weight)
        
        # Learnable scaling
        self.scales = nn.ParameterDict({
            str(i): nn.Parameter(torch.ones(1) * 0.1) for i in self.lora_layers
        })
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ LoRA initialized")
        print(f"  Layers: {self.lora_layers}")
        print(f"  Rank: {rank}, Alpha: {alpha}")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    
    def forward(self, x, layer_idx):
        """
        Apply LoRA adaptation to features.
        
        Args:
            x: (B, N, D) features from transformer block
            layer_idx: current layer index
        Returns:
            (B, N, D) adapted features
        """
        if layer_idx in self.lora_layers and str(layer_idx) in self.lora_modules:
            lora_out = self.lora_modules[str(layer_idx)](x)
            scale = self.scales[str(layer_idx)]
            return x + scale * lora_out
        return x


class LoRAConfig:
    """Configuration for LoRA experiments"""
    def __init__(self, lora_layers, rank=4, alpha=1.0):
        self.lora_layers = lora_layers
        self.rank = rank
        self.alpha = alpha
    
    def __repr__(self):
        return f"LoRAConfig(layers={self.lora_layers}, rank={self.rank}, alpha={self.alpha})"

