import pytest
import torch
from torch.testing import assert_close

from convnext.block import ConvNextBlock2d as ConvNextBlock2dBaseline


try:
    from convnext.te.block import ConvNextBlock2d
except ImportError:
    pytest.skip("Transformer Engine is not installed", allow_module_level=True)


class TestConvNextBlock2dTransformerEngine:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, normalization):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
        ).to("cuda")
        x = torch.randn(1, 32, 64, 64, device="cuda")
        with torch.autocast(device_type="cuda", dtype=dtype):
            y = block(x)
        assert x.shape == y.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("checkpoint", [True, False])
    def test_backward(self, dtype, normalization, checkpoint):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            checkpoint=checkpoint,
        ).to("cuda")
        x = torch.randn(1, 32, 64, 64, device="cuda")
        with torch.autocast(device_type="cuda", dtype=dtype):
            y = block(x)
        y.sum().backward()
        for param in block.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_baseline(self, normalization):
        baseline = ConvNextBlock2dBaseline(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
        ).to("cuda")
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
        ).to("cuda")

        block.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            block.get_parameter(name).data.copy_(param.data)

        x = torch.randn(1, 32, 64, 64, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = block(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline, atol=1e-4, rtol=0)
