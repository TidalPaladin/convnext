from typing import TYPE_CHECKING, Sequence, Tuple, cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..block import grid_to_tokens, tokens_to_grid
from ..drop_path import drop_path
from ..helpers import check_te_installed, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        check_te_installed(te)
        self.norm = te.LayerNorm(num_features, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        y = grid_to_tokens(x)
        y = self.norm(y)
        return tokens_to_grid(y, x.shape[2:])


class RMSNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        check_te_installed(te)
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
        bias: bool = True,
        drop_path_rate: float = 0.0,
        checkpoint: bool = False,
    ):
        super().__init__()
        check_te_installed(te)
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
            bias=bias,
        )

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
