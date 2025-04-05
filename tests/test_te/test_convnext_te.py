from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from convnext.convnext import ConvNextConfig


try:
    from convnext.te.convnext import ConvNext2d
except ImportError:
    pytest.skip("Transformer Engine is not installed", allow_module_level=True)


@pytest.fixture
def config():
    config = ConvNextConfig(
        in_channels=3,
        depths=(2, 2, 2),
        hidden_sizes=[64, 96, 128],
        ffn_hidden_sizes=[256, 384, 512],
        patch_size=(4, 4),
        kernel_size=(3, 3),
        activation="srelu",
        normalization="LayerNorm",
        hidden_dropout=0.1,
        drop_path_rate=0.1,
        backend="te",
    )
    return config


class TestConvNext:

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    up_depths=(2, 2, 2),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    def test_forward(self, config, exp):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, device="cuda")
        model = ConvNext2d(config).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        assert out.shape == exp

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    up_depths=(2, 2, 2),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, exp, checkpoint):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        config = replace(config, checkpoint=checkpoint)
        model = ConvNext2d(config).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        out.sum().backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_fewer_up_levels(self):
        config = ConvNextConfig(
            in_channels=1,
            depths=(2, 2, 2),
            up_depths=(2, 2),
            hidden_sizes=(32, 48, 64),
            ffn_hidden_sizes=(128, 192, 256),
            patch_size=(4, 4),
            kernel_size=(3, 3),
        )

        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, device="cuda")
        model = ConvNext2d(config).to("cuda")
        out = model(x)
        assert out.shape == (1, 48, 32, 32)

    @pytest.mark.parametrize("pool_type", ["avg", None])
    def test_forward_head(self, config, pool_type):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ConvNext2d(config).to("cuda")
        head = model.create_head(out_dim=10, pool_type=pool_type).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            features = model(x)
            out = head(features)
        exp = (1, 10) if pool_type is not None else (1, 10, 14, 14)
        assert out.shape == exp

    def test_baseline(self, config):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ConvNext2d(config).to("cuda")
        baseline = ConvNext2d(replace(config, backend="pytorch")).to("cuda")

        for name, param in model.named_parameters():
            baseline.get_parameter(name).data.copy_(param.data)

        model.eval()
        baseline.eval()

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            features = model(x)
            features_baseline = baseline(x)
        assert_close(features, features_baseline, atol=1e-4, rtol=0)
