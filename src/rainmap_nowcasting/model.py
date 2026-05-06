from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RainMapUNet(nn.Module):
    """Small U-Net that predicts future grayscale rain map frames."""

    def __init__(self, input_frames: int = 6, target_frames: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.base_channels = base_channels

        self.enc1 = ConvBlock(input_frames, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, target_frames, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x = _pad_to_multiple(x, multiple=8)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([_match_size(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([_match_size(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([_match_size(d1, e1), e1], dim=1))
        y = torch.sigmoid(self.out(d1))
        return y[..., : original_size[0], : original_size[1]]


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return x[..., : reference.shape[-2], : reference.shape[-1]]


def build_model_from_config(config: dict) -> RainMapUNet:
    return RainMapUNet(
        input_frames=int(config.get("input_frames", 6)),
        target_frames=int(config.get("target_frames", 6)),
        base_channels=int(config.get("base_channels", 32)),
    )
