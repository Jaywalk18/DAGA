"""
VPT: Visual Prompt Tuning for Vision Transformers
Based on: "Visual Prompt Tuning" (ECCV 2022)

VPT-Deep: Learnable prompts inserted at every transformer layer
"""
import torch
import torch.nn as nn
import math


class VPTDeep(nn.Module):
    """
    VPT-Deep: Visual Prompt Tuning with deep prompts.
    Inserts learnable prompt tokens at specified transformer layers.
    """
    def __init__(self, embed_dim=768, num_prompts=10, vpt_layers=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        self.vpt_layers = vpt_layers if vpt_layers else []
        
        # Learnable prompt tokens for each layer
        self.prompts = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            for i in self.vpt_layers
        })
        
        # Initialize prompts
        for prompt in self.prompts.values():
            nn.init.uniform_(prompt, -0.02, 0.02)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ VPT-Deep initialized")
        print(f"  Layers: {self.vpt_layers}")
        print(f"  Num prompts per layer: {num_prompts}")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    
    def get_prompts(self, layer_idx, batch_size):
        """
        Get prompt tokens for a specific layer.
        
        Args:
            layer_idx: transformer layer index
            batch_size: current batch size
        Returns:
            (B, num_prompts, D) prompt tokens or None
        """
        if layer_idx in self.vpt_layers and str(layer_idx) in self.prompts:
            prompts = self.prompts[str(layer_idx)]
            prompts = prompts.expand(batch_size, -1, -1)
            return self.dropout(prompts)
        return None
    
    def insert_prompts(self, x, layer_idx):
        """
        Insert prompt tokens into the sequence.
        
        Args:
            x: (B, N, D) input features (CLS + registers + patches)
            layer_idx: current layer index
        Returns:
            (B, N+num_prompts, D) features with prompts inserted
        """
        if layer_idx not in self.vpt_layers:
            return x, 0
        
        B = x.shape[0]
        prompts = self.get_prompts(layer_idx, B)
        
        if prompts is None:
            return x, 0
        
        # Insert prompts after CLS token
        # x: [CLS, registers, patches] -> [CLS, prompts, registers, patches]
        cls_token = x[:, :1, :]
        rest = x[:, 1:, :]
        
        x_with_prompts = torch.cat([cls_token, prompts, rest], dim=1)
        return x_with_prompts, self.num_prompts
    
    def remove_prompts(self, x, num_prompts_inserted):
        """
        Remove prompt tokens from the sequence.
        
        Args:
            x: (B, N+num_prompts, D) features with prompts
            num_prompts_inserted: number of prompts to remove
        Returns:
            (B, N, D) features without prompts
        """
        if num_prompts_inserted == 0:
            return x
        
        cls_token = x[:, :1, :]
        rest = x[:, 1+num_prompts_inserted:, :]
        return torch.cat([cls_token, rest], dim=1)


class VPTShallow(nn.Module):
    """
    VPT-Shallow: Visual Prompt Tuning with shallow prompts.
    Inserts learnable prompts only at the input layer.
    """
    def __init__(self, embed_dim=768, num_prompts=10, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        
        # Learnable prompt tokens (only for input)
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.uniform_(self.prompts, -0.02, 0.02)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ VPT-Shallow initialized")
        print(f"  Num prompts: {num_prompts}")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    
    def forward(self, x):
        """
        Insert prompts at input.
        
        Args:
            x: (B, N, D) input features
        Returns:
            (B, N+num_prompts, D) features with prompts
        """
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)
        prompts = self.dropout(prompts)
        
        cls_token = x[:, :1, :]
        rest = x[:, 1:, :]
        
        return torch.cat([cls_token, prompts, rest], dim=1)


class VPTConfig:
    """Configuration for VPT experiments"""
    def __init__(self, vpt_layers, num_prompts=10, dropout=0.0, mode='deep'):
        self.vpt_layers = vpt_layers
        self.num_prompts = num_prompts
        self.dropout = dropout
        self.mode = mode  # 'deep' or 'shallow'
    
    def __repr__(self):
        return (f"VPTConfig(mode={self.mode}, layers={self.vpt_layers}, "
                f"num_prompts={self.num_prompts})")

