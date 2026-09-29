"""
ViT-Adapter: Lightweight adapter for Vision Transformers
Based on: "Vision Transformer Adapter for Dense Predictions" (NeurIPS 2022)
"""
import torch
import torch.nn as nn


class SpatialPriorModule(nn.Module):
    """Spatial Prior Module - lightweight convolutional adapter"""
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        """Extract multi-scale spatial priors"""
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        
        return c1, c2, c3, c4


class Adapter(nn.Module):
    """Lightweight adapter module inserted after transformer blocks"""
    def __init__(self, embed_dim=768, mlp_ratio=0.25, drop=0.):
        super().__init__()
        self.down_sample = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()
        self.up_sample = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.drop = nn.Dropout(drop)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # Learnable scale
        
        # Initialize with small values
        nn.init.xavier_uniform_(self.down_sample.weight, gain=0.01)
        nn.init.zeros_(self.down_sample.bias)
        nn.init.zeros_(self.up_sample.weight)
        nn.init.zeros_(self.up_sample.bias)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) transformer features
        Returns:
            adapted features (B, N, D)
        """
        identity = x
        x = self.down_sample(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.up_sample(x)
        x = self.drop(x)
        return identity + self.scale * x


class ViTAdapter(nn.Module):
    """
    ViT-Adapter for parameter-efficient fine-tuning
    Paper: Vision Transformer Adapter for Dense Predictions (NeurIPS 2022)
    """
    def __init__(self, embed_dim=768, adapter_layers=None, mlp_ratio=0.25):
        super().__init__()
        self.adapter_layers = adapter_layers if adapter_layers else []
        
        # Create adapter for specified layers
        self.adapters = nn.ModuleDict({
            str(i): Adapter(embed_dim=embed_dim, mlp_ratio=mlp_ratio)
            for i in self.adapter_layers
        })
        
        print(f"✓ ViT-Adapter initialized for layers {self.adapter_layers}")
        print(f"  Adapter MLP ratio: {mlp_ratio}")
        print(f"  Embed dim: {embed_dim}")

    def forward(self, features, layer_idx):
        """
        Apply adapter to features from specific layer
        Args:
            features: (B, N, D) features from transformer block
            layer_idx: which layer this is
        Returns:
            adapted features (B, N, D)
        """
        if layer_idx in self.adapter_layers and str(layer_idx) in self.adapters:
            return self.adapters[str(layer_idx)](features)
        return features

