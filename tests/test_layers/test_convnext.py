from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from convnext.convnext import ConvNext2d, ConvNextConfig
from convnext.helpers import try_import_te


te = try_import_te()


@pytest.fixture(params=["pytorch", pytest.param("te", marks=pytest.mark.cuda)])
def config(request):
    config = ConvNextConfig(
        in_channels=3,
        depths=(2, 2, 2),
        hidden_sizes=[64, 96, 128],
        ffn_hidden_sizes=[256, 384, 512],
        patch_size=(4, 4),
        kernel_size=(3, 3),
        activation="srelu",
        normalization="LayerNorm",
        drop_path_rate=0.1,
        backend=request.param,
    )
    return config


class TestConvNext:

    def test_config_from_yaml_str(self, config):
        config_str = config.to_yaml()
        config_from_str = ConvNextConfig.from_yaml(config_str)
        assert config == config_from_str

    def test_config_from_yaml_path(self, config, tmp_path):
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            f.write(config.to_yaml())
        config_from_path = ConvNextConfig.from_yaml(path)
        assert config == config_from_path

    @pytest.mark.parametrize(
        "backend",
        [
            "pytorch",
            pytest.param("te", marks=pytest.mark.cuda),
        ],
    )
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
    def test_forward(self, config, backend, exp):
        config = replace(config, backend=backend)
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, device=config.device_type)
        model = ConvNext2d(config).to(config.device_type)
        with torch.autocast(device_type=model.device_type, dtype=torch.bfloat16, enabled=True):
            out = model(x)
        assert out.shape == exp

    @pytest.mark.parametrize(
        "backend",
        [
            "pytorch",
            pytest.param("te", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "config",
        [
            ConvNextConfig(
                in_channels=1,
                depths=(2, 2, 2),
                hidden_sizes=(32, 48, 64),
                ffn_hidden_sizes=(128, 192, 256),
                patch_size=(4, 4),
                kernel_size=(3, 3),
            ),
            ConvNextConfig(
                in_channels=1,
                depths=(2, 2, 2),
                hidden_sizes=(32, 48, 64),
                ffn_hidden_sizes=(128, 192, 256),
                up_depths=(2, 2, 2),
                patch_size=(4, 4),
                kernel_size=(3, 3),
            ),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, backend, checkpoint):
        config = replace(config, backend=backend)
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, requires_grad=True, device=config.device_type)
        config = replace(config, checkpoint=checkpoint)
        model = ConvNext2d(config).to(config.device_type)
        with torch.autocast(device_type=model.device_type, dtype=torch.bfloat16, enabled=True):
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
        x = torch.randn(B, C, H, W, device=config.device_type)
        model = ConvNext2d(config).to(config.device_type)
        out = model(x)
        assert out.shape == (1, 48, 32, 32)

    @pytest.mark.parametrize("pool_type", ["avg", None])
    def test_forward_head(self, config, pool_type):
        x = torch.randn(1, 3, 224, 224, device=config.device_type)
        model = ConvNext2d(config).to(config.device_type)
        head = model.create_head(out_dim=10, pool_type=pool_type).to(config.device_type)
        with torch.autocast(device_type=model.device_type, dtype=torch.bfloat16, enabled=True):
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
        x = torch.randn(B, C, H, W, device=config1.device_type)
        model1 = ConvNext2d(config1).to(config1.device_type)
        model2 = ConvNext2d(config2).to(config2.device_type)
        with torch.autocast(device_type=model1.device_type, dtype=torch.bfloat16, enabled=True):
            out1 = model1(x)
            out2 = model2(x)
        assert out1.shape == (1, 72, 16, 16)
        assert out2.shape == (1, 48, 64, 64)

    @pytest.mark.cuda
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_baseline(self, config, normalization):
        config = ConvNextConfig(
            in_channels=3,
            depths=(3, 3, 9, 3),
            hidden_sizes=[64, 96, 128, 256],
            ffn_hidden_sizes=[256, 384, 512, 1024],
            patch_size=(4, 4),
            kernel_size=(3, 3),
            activation="srelu",
            normalization=normalization,
            drop_path_rate=0.1,
            backend="pytorch",
        )
        te_config = replace(config, backend="te")

        x = torch.randn(1, 3, 224, 224)
        model = ConvNext2d(config)
        baseline = ConvNext2d(te_config).to("cuda")

        for name, param in model.named_parameters():
            baseline.get_parameter(name).data.copy_(param.data)

        model.eval()
        baseline.eval()

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            features = model(x)
            features_baseline = baseline(x.cuda())
        assert_close(features, features_baseline, atol=1e-3, rtol=0, check_device=False)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    @pytest.mark.parametrize("mlp", [False, True])
    def test_forward_head_no_pooling(self, backend, config, mlp):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend)
        device = "cuda" if backend == "te" else "cpu"

        x = torch.randn(1, 3, 224, 224, device=device)
        model = ConvNext2d(config).to(device)
        head = model.create_head(1, mlp=mlp)
        head = head.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            out = model(x)
            out = head(out)
        assert out.shape == (1, 1, 14, 14)

    @pytest.mark.parametrize("backend", ["pytorch", "te"])
    @pytest.mark.parametrize("mlp", [False, True])
    @pytest.mark.parametrize("pool_type", ["avg", "max"])
    def test_forward_head_pooling(self, backend, config, mlp, pool_type):
        if backend == "te" and te is None:
            pytest.skip("Transformer Engine is not available")
        config = replace(config, backend=backend)
        device = "cuda" if backend == "te" else "cpu"

        x = torch.randn(1, 3, 224, 224, device=device)
        model = ConvNext2d(config).to(device)
        head = model.create_head(1, mlp=mlp, pool_type=pool_type)
        head = head.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            out = model(x)
            out = head(out)
        assert out.shape == (1, 1)
