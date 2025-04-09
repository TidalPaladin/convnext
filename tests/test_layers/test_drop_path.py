import torch
from torch.testing import assert_close

from convnext.drop_path import DropPath


class TestDropPath:

    def test_forward(self):
        torch.random.manual_seed(42)
        drop_prob = 0.1
        drop_path = DropPath(drop_prob=drop_prob)
        B, C, H, W = 32, 8, 16, 16
        x = torch.randn(B, C, H, W)
        y = drop_path(x)
        assert_close(y[0], x[0] / (1 - drop_prob))

    def test_forward_large_sample(self):
        torch.random.manual_seed(42)
        drop_prob = 0.1
        drop_path = DropPath(drop_prob=drop_prob)
        B, C, H, W = 8192, 8, 16, 16
        x = torch.randn(B, C, H, W)
        is_dropped = drop_path(x).view(B, -1).eq(0).all(dim=-1).float().mean()
        TOL = 0.01
        assert drop_prob - TOL <= is_dropped <= drop_prob + TOL

    def test_forward_eval(self):
        torch.random.manual_seed(42)
        drop_prob = 0.1
        drop_path = DropPath(drop_prob=drop_prob)
        drop_path.eval()
        B, C, H, W = 32, 8, 16, 16
        x = torch.randn(B, C, H, W)
        y = drop_path(x)
        assert_close(y, x)
