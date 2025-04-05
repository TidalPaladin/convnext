# ConvNext

Implementation of ConvNext that supports both native PyTorch and Transformer Engine backends.

## Installation

The PyTorch backend is easily installable like a normal repository

```bash
pip install convnext @ git+https://github.com/TidalPaladin/convnext.git
```

To enable the ConvNext backend, manually install Transformer Engine with PyTorch support

```bash
pip install --no-build-isolation "transformer-engine[pytorch]"
```

The Transformer Engine backend is optional, and an `ImportError` will be raised when passing
`backend='te'` without installing Transformer Engine.

## Usage

```python
from convnext import ConvNextConfig

config = ConvNextConfig(
    in_channels=3,
    depths=(2, 2, 2),
    #up_depths=(2, 2) # U-Net design
    hidden_sizes=[64, 96, 128],
    ffn_hidden_sizes=[256, 384, 512],
    patch_size=(4, 4),
    kernel_size=(7, 7),
    activation="srelu",
    normalization="LayerNorm",
    hidden_dropout=0.1,
    drop_path_rate=0.1,
    backend="pytorch", # or 'te' for Transformer Engine
)
model = config.instantiate()

B, C, H, W = 2, 3, 224, 224
x = torch.rand(B, C, H W)
y = model(x)
```