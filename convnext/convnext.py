from dataclasses import dataclass, field
from typing import ClassVar, List, Sequence, Tuple, Type, cast

import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from .block import ConvNextBlock2d, LayerNorm2d, grid_to_tokens, tokens_to_grid
from .helpers import TRUNC_NORMAL_STD


@dataclass(frozen=True)
class ConvNextConfig:
    # Inputs
    in_channels: int
    patch_size: Sequence[int]

    # ConvNext Blocks
    kernel_size: Sequence[int]
    depths: Sequence[int]
    hidden_sizes: Sequence[int]
    ffn_hidden_sizes: Sequence[int]
    normalization: str = "LayerNorm"
    bias: bool = True
    activation: str = "srelu"
    hidden_dropout: float = 0.0
    drop_path_rate: float = 0.0
    checkpoint: bool = False

    # Optional U-Net style upsampling
    up_depths: Sequence[int] = field(default_factory=lambda: [])

    def instantiate(self) -> "ConvNext2d":
        return ConvNext2d(self)

    @property
    def isotropic_output_dim(self) -> int:
        if self.up_depths:
            return list(reversed(self.hidden_sizes))[len(self.up_depths) - 1]
        return self.hidden_sizes[-1]


class ConvNext2d(nn.Module):
    CONFIG_TYPE: ClassVar[Type[ConvNextConfig]] = ConvNextConfig

    def __init__(self, config: ConvNextConfig):
        super().__init__()
        self.config = config

        # Patch embedding stem
        self.stem = nn.Conv2d(
            self.config.in_channels,
            self.config.hidden_sizes[0],
            kernel_size=cast(Tuple[int, int], tuple(self.config.patch_size)),
            stride=cast(Tuple[int, int], tuple(self.config.patch_size)),
        )
        self.norm = nn.LayerNorm(self.config.hidden_sizes[0])

        # Down stages
        self.down_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock2d(
                            self.config.hidden_sizes[i],
                            self.config.ffn_hidden_sizes[i],
                            kernel_size=cast(Tuple[int, int], tuple(self.config.kernel_size)),
                            activation=self.config.activation,
                            normalization=self.config.normalization,
                            checkpoint=self.config.checkpoint,
                            dropout=self.config.hidden_dropout,
                            drop_path_rate=self.config.drop_path_rate,
                        )
                        for _ in range(self.config.depths[i])
                    ]
                )
                for i in range(len(self.config.depths))
            ]
        )
        # Downsampling blocks at end of each stage
        self.downsample = nn.ModuleList(
            [
                nn.Conv2d(self.config.hidden_sizes[i], self.config.hidden_sizes[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.depths) - 1)
            ]
        )

        # Up stages
        up_dims = list(reversed(self.config.hidden_sizes))
        up_dims_feedforward = list(reversed(self.config.ffn_hidden_sizes))
        self.up_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock2d(
                            up_dims[i],
                            up_dims_feedforward[i],
                            kernel_size=cast(Tuple[int, int], tuple(self.config.kernel_size)),
                            activation=self.config.activation,
                            normalization=self.config.normalization,
                            checkpoint=self.config.checkpoint,
                            dropout=self.config.hidden_dropout,
                            drop_path_rate=self.config.drop_path_rate,
                        )
                        for _ in range(self.config.up_depths[i])
                    ]
                )
                for i in range(len(self.config.up_depths) - 1)
            ]
        )

        # Upsampling blocks at end of each stage
        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose2d(up_dims[i], up_dims[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.up_depths) - 1)
            ]
        )

    def create_head(
        self,
        out_dim: int,
        pool_type: str | None = None,
    ) -> nn.Module:
        r"""Creates a head for the model.

        Args:
            out_dim: Dimension of the output.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.
        """
        layer = nn.Sequential()

        # Pooling type
        match pool_type:
            case "avg":
                layer.add_module("pool", nn.AdaptiveAvgPool2d((1, 1)))
                layer.add_module("reshape", Rearrange("b c 1 1 -> b c"))
            case "max":
                layer.add_module("pool", nn.AdaptiveMaxPool2d((1, 1)))
                layer.add_module("reshape", Rearrange("b c 1 1 -> b c"))
            case None:
                pass
            case _:
                raise ValueError(f"Invalid pool type: {pool_type}")

        # Normalization + Linear
        if pool_type is not None:
            layer.add_module("norm", nn.LayerNorm(self.config.isotropic_output_dim))
            linear = nn.Linear(self.config.isotropic_output_dim, out_dim)
            nn.init.trunc_normal_(linear.weight, std=TRUNC_NORMAL_STD)
            layer.add_module("linear", linear)
        else:
            layer.add_module("norm", LayerNorm2d(self.config.isotropic_output_dim))
            conv = nn.Conv2d(self.config.isotropic_output_dim, out_dim, kernel_size=1)
            layer.add_module("conv", conv)

        return layer

    def forward(self, x: Tensor) -> Tensor:
        # Patch embed stem
        x = self.stem(x)
        x = tokens_to_grid(self.norm(grid_to_tokens(x)), x.shape[2:])

        # Run down blocks
        levels: List[Tensor] = []
        for i, stage in enumerate(self.down_stages):
            for block in cast(nn.ModuleList, stage):
                x = block(x)
            levels.append(x)

            # Downsample and verify new size
            if i < len(self.downsample):
                x = self.downsample[i](x)

        levels.pop(-1)

        # Run up blocks
        for i, stage in enumerate(self.up_stages):
            # Process current level
            for block in cast(nn.ModuleList, stage):
                x = block(x)

            # Upsample and verify new size
            if i < len(self.upsample):
                x = self.upsample[i](x)

            # Add with previous level
            y = levels[-(i + 1)]
            x = x + y
            levels[-(i + 1)] = x

        return x
