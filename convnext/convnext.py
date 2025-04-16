from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Self, Sequence, Tuple, Type, cast

import torch.nn as nn
import yaml
from einops.layers.torch import Rearrange
from torch import Tensor

from .block import ConvNextBlock2d, LayerNorm2d, RMSNorm2d, grid_to_tokens, tokens_to_grid
from .helpers import TRUNC_NORMAL_STD, check_te_installed, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


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
    drop_path_rate: float = 0.0
    checkpoint: bool = False

    # Optional U-Net style upsampling
    up_depths: Sequence[int] = field(default_factory=lambda: [])

    # Backend selection
    backend: Literal["pytorch", "te"] = "pytorch"

    def instantiate(self) -> "ConvNext2d":
        if self.backend not in ["pytorch", "te"]:
            raise ValueError(f"Invalid backend: {self.backend}")
        return ConvNext2d(self)

    @property
    def device_type(self) -> str:
        return "cuda" if self.backend == "te" else "cpu"

    @property
    def isotropic_output_dim(self) -> int:
        if self.up_depths:
            return list(reversed(self.hidden_sizes))[len(self.up_depths) - 1]
        return self.hidden_sizes[-1]

    @property
    def block_kwargs(self) -> Dict[str, Any]:
        return dict(
            kernel_size=tuple(self.kernel_size),
            activation=self.activation,
            normalization=self.normalization,
            checkpoint=self.checkpoint,
            drop_path_rate=self.drop_path_rate,
            backend=self.backend,
        )

    @classmethod
    def from_yaml(cls: Type[Self], path: str | Path) -> Self:
        if isinstance(path, Path):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                config = yaml.full_load(f)
            return cls(**config)

        elif isinstance(path, str) and path.endswith(".yaml"):
            return cls.from_yaml(Path(path))

        else:
            config = yaml.full_load(path)
            return cls(**config)

    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)


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
        self.norm = self.create_norm(self.config.hidden_sizes[0])

        # Down stages
        self.down_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self.create_block(self.config.hidden_sizes[i], self.config.ffn_hidden_sizes[i])
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
                    [self.create_block(up_dims[i], up_dims_feedforward[i]) for _ in range(self.config.up_depths[i])]
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

    @property
    def device_type(self) -> str:
        return self.config.device_type

    def create_block(self, hidden_size: int, ffn_hidden_size: int) -> nn.Module:
        return ConvNextBlock2d(hidden_size, ffn_hidden_size, **self.config.block_kwargs)

    def create_norm(self, dim: int) -> nn.Module:
        match (self.config.normalization, self.config.backend):
            case ("LayerNorm", "pytorch"):
                return nn.LayerNorm(dim, eps=1e-5)
            case ("LayerNorm", "te"):
                check_te_installed(te)
                return te.LayerNorm(dim, eps=1e-5)
            case ("RMSNorm", "pytorch"):
                return nn.RMSNorm(dim, eps=1e-5)
            case ("RMSNorm", "te"):
                check_te_installed(te)
                return te.RMSNorm(dim, eps=1e-5)
            case _:
                raise ValueError(
                    f"Invalid normalization: {self.config.normalization} for backend: {self.config.backend}"
                )

    def create_norm_2d(self, dim: int) -> nn.Module:
        match self.config.normalization:
            case "LayerNorm":
                return LayerNorm2d(dim, backend=self.config.backend, eps=1e-5)
            case "RMSNorm":
                return RMSNorm2d(dim, backend=self.config.backend, eps=1e-5)
            case _:
                raise ValueError(f"Invalid normalization: {self.config.normalization}")

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
            layer.add_module("norm", self.create_norm(self.config.isotropic_output_dim))
            if self.config.backend == "pytorch":
                linear = nn.Linear(self.config.isotropic_output_dim, out_dim)
                nn.init.trunc_normal_(linear.weight, std=TRUNC_NORMAL_STD)
            elif self.config.backend == "te":
                check_te_installed(te)
                linear = te.Linear(self.config.isotropic_output_dim, out_dim)
            else:
                raise ValueError(f"Invalid backend: {self.config.backend}")
            layer.add_module("linear", linear)
        else:
            layer.add_module("norm", self.create_norm_2d(self.config.isotropic_output_dim))
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
