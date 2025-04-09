from typing import Sequence, Tuple, cast

import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
from torch import Tensor

from ..block import grid_to_tokens, tokens_to_grid
from ..drop_path import drop_path


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = te.LayerNorm(num_features, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        y = grid_to_tokens(x)
        y = self.norm(y)
        return tokens_to_grid(y, x.shape[2:])


class RMSNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = te.RMSNorm(num_features, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        y = grid_to_tokens(x)
        y = self.norm(y)
        return tokens_to_grid(y, x.shape[2:])


class ConvNextBlock2d(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        kernel_size: Sequence[int] = (7, 7),
        activation: str = "srelu",
        normalization: str = "LayerNorm",
        stride: Sequence[int] = (1, 1),
        dropout: float = 0.0,
        bias: bool = True,
        drop_path_rate: float = 0.0,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.checkpoint = checkpoint

        # Depthwise convolution
        padding = [(k - s) // 2 for k, s in zip(kernel_size, stride)]
        self.conv_dw = nn.Conv2d(
            hidden_size,
            hidden_size,
            cast(Tuple[int, int], kernel_size),
            stride=cast(Tuple[int, int], stride),
            padding=cast(Tuple[int, int], padding),
            groups=hidden_size,
        )

        # MLP
        self.mlp = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            activation=activation,
            normalization=normalization,
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv_dw.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        y = F.conv2d(
            x,
            self.conv_dw.weight,
            self.conv_dw.bias,
            stride=self.conv_dw.stride,
            padding=self.conv_dw.padding,
            groups=self.conv_dw.groups,
        )
        y = grid_to_tokens(y)
        y = self.mlp(y)
        y = tokens_to_grid(y, x.shape[2:])
        return x + drop_path(y, self.drop_path_rate, self.training)
