import pytest
import torch
from torch.testing import assert_close

from convnext.block import ConvNextBlock2d


@pytest.fixture(params=["pytorch", pytest.param("te", marks=pytest.mark.cuda)])
def backend(request):
    return request.param


class TestConvNextBlock2d:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, normalization, backend):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
            backend=backend,
        )
        device_type = "cuda" if backend == "te" else "cpu"
        block = block.to(device_type)
        x = torch.randn(1, 32, 64, 64, device=device_type)
        with torch.autocast(device_type=device_type, dtype=dtype):
            y = block(x)
        assert x.shape == y.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("checkpoint", [True, False])
    def test_backward(self, dtype, normalization, checkpoint, backend):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            checkpoint=checkpoint,
            backend=backend,
        )
        device_type = "cuda" if backend == "te" else "cpu"
        block = block.to(device_type)
        x = torch.randn(1, 32, 64, 64, device=device_type)
        with torch.autocast(device_type=device_type, dtype=dtype):
            y = block(x)
        y.sum().backward()
        for param in block.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    def test_deterministic(self, backend):
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            backend=backend,
        )
        device_type = "cuda" if backend == "te" else "cpu"
        block = block.to(device_type)
        x = torch.randn(1, 32, 64, 64, device=device_type)
        block.eval()
        y3 = block(x)
        y4 = block(x)
        assert torch.allclose(y3, y4)

    @pytest.mark.cuda
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_baseline(self, normalization):
        baseline = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
            backend="pytorch",
        ).to("cuda")
        block = ConvNextBlock2d(
            hidden_size=32,
            ffn_hidden_size=64,
            normalization=normalization,
            drop_path_rate=0.1,
            backend="te",
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
