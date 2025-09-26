"""U-Net-style SuperPoint variant with deeper encoder blocks."""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two stacked conv-batchnorm-ReLU layers used by the encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InitialConv(nn.Module):
    """Initial double convolution applied before the downsampling path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """Max-pooling followed by a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class SuperPointNetGauss2(nn.Module):
    """U-Net-inspired SuperPoint backbone with batch-normalized detector head."""

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        self.relu = nn.ReLU(inplace=True)
        self.inc = InitialConv(1, c1)
        self.down1 = DownsampleBlock(c1, c2)
        self.down2 = DownsampleBlock(c2, c3)
        self.down3 = DownsampleBlock(c3, c4)

        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(65)

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning detector logits and normalized descriptors.

        Args:
            x: Input image tensor with shape ``(batch, 1, height, width)``.

        Returns:
            A tuple ``(semi, desc)`` where ``semi`` contains detector logits and
            ``desc`` stores L2-normalized descriptors.
        """
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))

        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))
        desc = F.normalize(desc, p=2, dim=1, eps=1e-8)

        return semi, desc


__all__ = ["SuperPointNetGauss2"]
