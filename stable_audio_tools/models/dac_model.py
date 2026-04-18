"""
Local copy of the DAC encoder/decoder modules.

Originally adapted from: https://github.com/descriptinc/descript-audio-codec
(`dac.model.dac` and `dac.nn.layers`).

Adds an optional `use_rmsnorm` flag that inserts RMSNorm layers throughout
the residual units, encoder blocks, and decoder blocks (matching the
reference implementation provided by the user).
"""

import math
from typing import List

import torch
from torch import nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class RMSNorm(nn.Module):
    """RMSNorm over the channel dimension for [B, C, T] tensors."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=1, keepdim=True) + self.eps)
        return ((x_float / rms) * self.gamma).to(x.dtype)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, use_rmsnorm: bool = False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, use_rmsnorm: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, use_rmsnorm=use_rmsnorm),
            ResidualUnit(dim // 2, dilation=3, use_rmsnorm=use_rmsnorm),
            ResidualUnit(dim // 2, dilation=9, use_rmsnorm=use_rmsnorm),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, use_rmsnorm=use_rmsnorm)]

        self.block += [
            RMSNorm(d_model) if use_rmsnorm else nn.Identity(),
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, use_rmsnorm: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=3, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=9, use_rmsnorm=use_rmsnorm),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        use_rmsnorm: bool = False,
    ):
        super().__init__()

        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, use_rmsnorm=use_rmsnorm)]

        layers += [
            RMSNorm(output_dim) if use_rmsnorm else nn.Identity(),
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
