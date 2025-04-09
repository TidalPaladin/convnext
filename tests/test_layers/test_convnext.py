from dataclasses import replace

import pytest
import torch

from convnext.convnext import ConvNext2d, ConvNextConfig


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
                    normalization="RMSNorm",
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
                    normalization="LayerNorm",
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    def test_forward(self, config, exp):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W)
        model = ConvNext2d(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
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
        x = torch.randn(B, C, H, W, requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ConvNext2d(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
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
        x = torch.randn(B, C, H, W)
        model = ConvNext2d(config)
        out = model(x)
        assert out.shape == (1, 48, 32, 32)

    @pytest.mark.parametrize("pool_type", ["avg", None])
    def test_forward_head(self, config, pool_type):
        x = torch.randn(1, 3, 224, 224)
        model = ConvNext2d(config)
        head = model.create_head(out_dim=10, pool_type=pool_type)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            features = model(x)
            out = head(features)
        exp = (1, 10) if pool_type is not None else (1, 10, 14, 14)
        assert out.shape == exp

    def test_unet_size(self):
        torch.random.manual_seed(42)
        # Coarsest level = 256/16 = 16
        config1 = ConvNextConfig(
            in_channels=1,
            depths=(2, 2, 2, 2),
            hidden_sizes=(32, 48, 64, 72),
            ffn_hidden_sizes=(128, 192, 256, 288),
            patch_size=(2, 2),
            kernel_size=(3, 3),
        )
        # Two up levels = 64
        config2 = replace(config1, up_depths=(2, 2, 2))

        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W)
        model1 = ConvNext2d(config1)
        model2 = ConvNext2d(config2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out1 = model1(x)
            out2 = model2(x)
        assert out1.shape == (1, 72, 16, 16)
        assert out2.shape == (1, 48, 64, 64)
