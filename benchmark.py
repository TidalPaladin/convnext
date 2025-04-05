from dataclasses import replace
from timeit import timeit

import torch
import torch.nn as nn

from convnext import ConvNextConfig


def forward(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        y = model(x)
    torch.cuda.synchronize()
    return y


def forward_backward(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        y = model(x)
    y.sum().backward()
    torch.cuda.synchronize()
    return y


def main():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    config = ConvNextConfig(
        in_channels=3,
        depths=(3, 3, 9, 3),
        hidden_sizes=[64, 96, 128, 256],
        ffn_hidden_sizes=[256, 384, 512, 1024],
        patch_size=(4, 4),
        kernel_size=(7, 7),
        activation="srelu",
        normalization="RMSNorm",
        hidden_dropout=0.1,
        drop_path_rate=0.1,
    )
    torch_model = config.instantiate().to("cuda")
    te_model = replace(config, backend="te").instantiate().to("cuda")

    B, C, H, W = 2, 3, 224, 224
    x = torch.rand(B, C, H, W, device="cuda")

    # Forward pass
    with torch.inference_mode():
        torch_model.eval()
        te_model.eval()

        # Warmup
        forward(torch_model, x)
        forward(te_model, x)

        torch_time = timeit(lambda: forward(torch_model, x), number=100) / 100
        te_time = timeit(lambda: forward(te_model, x), number=100) / 100

        print("Forward pass:")
        print(f"Torch time: {torch_time}")
        print(f"TE time: {te_time}")
        print(f"Speedup: {torch_time / te_time}")

    # Forward-backward pass
    torch_model.train()
    te_model.train()

    # Warmup
    x.requires_grad = True
    forward(torch_model, x)
    forward(te_model, x)

    torch_time = timeit(lambda: forward_backward(torch_model, x), number=100) / 100
    te_time = timeit(lambda: forward_backward(te_model, x), number=100) / 100

    print("\nForward-backward pass:")
    print(f"Torch time: {torch_time}")
    print(f"TE time: {te_time}")
    print(f"Speedup: {torch_time / te_time}")


if __name__ == "__main__":
    main()
