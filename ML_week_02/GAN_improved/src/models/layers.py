"""
Custom layers for GAN architectures.

Includes:
- Spectral Normalization wrapper
- MinibatchStdDev layer (from ProGAN/StyleGAN)
- Pixel Normalization
- Self-Attention layer
- Custom weight initialization utilities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Weight Initialization
# ============================================================================

def weights_init_normal(m: nn.Module) -> None:
    """Initialize weights from N(0, 0.02) — DCGAN convention."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def weights_init_xavier(m: nn.Module) -> None:
    """Xavier/Glorot initialization — better for deeper networks."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ============================================================================
# Spectral Normalization (convenience wrapper)
# ============================================================================

def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to all Conv and Linear layers."""
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.utils.spectral_norm(layer)
    return module


# ============================================================================
# MinibatchStdDev — from ProGAN / StyleGAN
# ============================================================================

class MinibatchStdDev(nn.Module):
    """
    Appends the std-dev of each spatial position across the batch
    as an extra feature map. Helps the discriminator detect mode collapse
    by providing batch-level statistics.
    """

    def __init__(self, group_size: int = 4) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        # Reshape → [G, M, C, H, W] where G = group_size, M = b // G
        y = x.view(group_size, -1, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)           # center
        y = (y ** 2).mean(dim=0)                       # variance
        y = (y + 1e-8).sqrt()                          # std-dev
        y = y.mean(dim=[1, 2, 3], keepdim=True)        # average over CHW
        y = y.repeat(group_size, 1, h, w)              # replicate for batch
        return torch.cat([x, y], dim=1)


# ============================================================================
# Pixel Normalization — StyleGAN technique
# ============================================================================

class PixelNorm(nn.Module):
    """Normalize feature vectors per pixel to unit length."""

    def forward(self, x: Tensor) -> Tensor:
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8).sqrt()


# ============================================================================
# Self-Attention — SAGAN style
# ============================================================================

class SelfAttention(nn.Module):
    """
    Self-attention mechanism from SAGAN (Zhang et al., 2019).
    Allows the generator/discriminator to model long-range dependencies.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)  # B × N × C'
        k = self.key(x).view(b, -1, h * w)                       # B × C' × N
        attn = F.softmax(torch.bmm(q, k), dim=-1)                # B × N × N
        v = self.value(x).view(b, -1, h * w)                     # B × C × N
        out = torch.bmm(v, attn.permute(0, 2, 1))                # B × C × N
        out = out.view(b, c, h, w)
        return self.gamma * out + x


# ============================================================================
# Conditional Batch Normalization
# ============================================================================

class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization.
    The scale (gamma) and shift (beta) are predicted from a class embedding,
    allowing the normalization to be class-conditioned.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gain = nn.Embedding(num_classes, num_features)
        self.bias = nn.Embedding(num_classes, num_features)
        # Initialize to identity transform
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: Feature maps [B, C, H, W]
            y: Class labels [B] (integer indices)
        """
        out = self.bn(x)
        gain = self.gain(y).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        bias = self.bias(y).unsqueeze(-1).unsqueeze(-1)
        return out * gain + bias
