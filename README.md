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
```


Alternatively, to create a U-Net style ConvNext model

```python
# Coarsest level = 256/16 = 16
# Three upsampling stages
config = ConvNextConfig(
    in_channels=3,
    depths=(2, 2, 2, 2),
    up_depths=(2, 2, 2),
    hidden_sizes=(32, 48, 64, 72),
    ffn_hidden_sizes=(128, 192, 256, 288),
    patch_size=(2, 2),
    kernel_size=(7, 7),
)

B, C, H, W = 2, 3, 256, 256
x = torch.rand(B, C, H W)
model = ConvNext2d(config)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
    out = model(x)
assert out.shape == (2, 48, 64, 64)
```