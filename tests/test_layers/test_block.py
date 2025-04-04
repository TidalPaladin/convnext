import pytest
import torch

from convnext.block import ConvNextBlock2d


class TestConvNextBlock2d:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, normalization):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            dropout=0.1,
            drop_path_rate=0.1,
        )
        x = torch.randn(1, 32, 64, 64)
        with torch.autocast(device_type="cpu", dtype=dtype):
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
        )
        x = torch.randn(1, 32, 64, 64)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = block(x)
        y.sum().backward()
        for param in block.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_deterministic(self):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            dropout=0.1,
        )
        x = torch.randn(1, 32, 64, 64)
        block.train()
        y1 = block(x)
        y2 = block(x)
        assert not torch.allclose(y1, y2)

        block.eval()
        y3 = block(x)
        y4 = block(x)
        assert torch.allclose(y3, y4)
