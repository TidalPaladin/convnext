import pytest


def pytest_runtest_setup(item):
    """Skip all tests in this directory if transformer_engine is not installed."""
    try:
        import transformer_engine.pytorch  # type: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("Transformer Engine is not installed")

    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("CUDA is not available")
