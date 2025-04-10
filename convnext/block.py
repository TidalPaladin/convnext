import math
from typing import TYPE_CHECKING, Callable, Literal, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .drop_path import drop_path
from .helpers import TRUNC_NORMAL_STD, check_te_installed, compile_is_disabled, get_activation, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def tokens_to_grid(x: Tensor, size: Sequence[int]) -> Tensor:
    r"""Convert a channel-last flat token sequence to a channel-first spatial grid.

    Args:
        x: The token sequence to convert to a grid.
        size: The size of the grid to convert to.

    Returns:
        The grid of tokens.

    Raises:
        ValueError: If the token length does not match the grid size.
    """
    _, L, _ = x.shape
    if L != math.prod(size):
        raise ValueError(f"Token length {L} does not match grid size {size}")

    if len(size) == 1:
        y = rearrange(x, "b l c -> b c l")
    elif len(size) == 2:
        y = rearrange(x, "b (h w) c -> b c h w", h=size[0], w=size[1])
    elif len(size) == 3:
        y = rearrange(x, "b (d h w) c -> b c d h w", d=size[0], h=size[1], w=size[2])
    else:
        raise ValueError(f"Invalid size: {size}")

    return y.contiguous()


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def grid_to_tokens(x: Tensor) -> Tensor:
    r"""Convert a channel-first spatial grid to a channel-last flat token sequence.

    Args:
        x: The grid to convert to a token sequence.

    Returns:
        The token sequence.
    """
    return rearrange(x, "b c ... -> b (...) c").contiguous()


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def forward_layer_norm_mlp(
    x: Tensor,
    normalization: str,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
) -> Tensor:
    # Normalization
    if normalization == "LayerNorm":
        y = F.layer_norm(x, x.shape[-1:], weight=norm_weight, bias=norm_bias, eps=eps)
    elif normalization == "RMSNorm":
        y = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)
    else:
        raise ValueError(f"Normalization {normalization} not supported")

    # FF1
    y = F.linear(y, fc1_weight, fc1_bias)
    y = activation(y)

    # FF2
    y = F.linear(y, fc2_weight, fc2_bias)

    return y


class LayerNormMLP(nn.Module):
    r"""MLP with pre-normalization, matching TransformerEngine's LayerNormMLP."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str,
        normalization: str,
        bias: bool = True,
        checkpoint: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.eps = eps

        # Normalization
        self.normalization = normalization
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        if self.normalization == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.layer_norm_bias = None

        # MLP
        self.fc1_weight = nn.Parameter(torch.empty(ffn_hidden_size, hidden_size))
        self.fc2_weight = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size))
        self.fc1_bias = nn.Parameter(torch.zeros(ffn_hidden_size)) if bias else None
        self.fc2_bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.act = get_activation(activation)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)
        nn.init.trunc_normal_(self.fc1_weight, std=TRUNC_NORMAL_STD)
        nn.init.trunc_normal_(self.fc2_weight, std=TRUNC_NORMAL_STD)
        if self.fc1_bias is not None:
            nn.init.zeros_(self.fc1_bias)
        if self.fc2_bias is not None:
            nn.init.zeros_(self.fc2_bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.checkpoint:
            y = checkpoint(
                forward_layer_norm_mlp,
                x,
                self.normalization,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                self.fc1_bias,
                self.fc2_weight,
                self.fc2_bias,
                self.act,
                self.eps,
                use_reentrant=False,
            )
        else:
            y = forward_layer_norm_mlp(
                x,
                self.normalization,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                self.fc1_bias,
                self.fc2_weight,
                self.fc2_bias,
                self.act,
                self.eps,
            )
        assert isinstance(y, Tensor)
        assert y.shape == x.shape
        return y


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, backend: str = "pytorch"):
        super().__init__()
        match backend:
            case "pytorch":
                self.norm = nn.LayerNorm(num_features, eps=eps)
            case "te":
                check_te_installed(te)
                self.norm = te.LayerNorm(num_features, eps=eps)
            case _:
                raise ValueError(f"Backend {backend} not supported")

    def forward(self, x: Tensor) -> Tensor:
        y = grid_to_tokens(x)
        y = self.norm(y)
        return tokens_to_grid(y, x.shape[2:])


class RMSNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, backend: str = "pytorch"):
        super().__init__()
        match backend:
            case "pytorch":
                self.norm = nn.RMSNorm(num_features, eps=eps)
            case "te":
                check_te_installed(te)
                self.norm = te.RMSNorm(num_features, eps=eps)
            case _:
                raise ValueError(f"Backend {backend} not supported")

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
        eps: float = 1e-5,
        backend: Literal["pytorch", "te"] = "pytorch",
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

        match backend:
            case "pytorch":
                self.mlp = LayerNormMLP(
                    hidden_size,
                    ffn_hidden_size,
                    activation,
                    normalization,
                    bias,
                    checkpoint,
                    eps,
                )
            case "te":
                check_te_installed(te)
                self.mlp = te.LayerNormMLP(
                    hidden_size,
                    ffn_hidden_size,
                    activation=activation,
                    normalization=normalization,
                    bias=bias,
                    eps=eps,
                )
            case _:
                raise ValueError(f"Backend {backend} not supported")

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
