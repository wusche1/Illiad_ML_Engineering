"""Test functions for PyTorch basics exercises."""
import torch
from einops import rearrange, reduce


def test_pairwise_distances(fn):
    print("Testing pairwise_distances...", end=" ")
    torch.manual_seed(0)
    points = torch.randn(20, 2)
    result = fn(points)

    if result.shape != (20, 20):
        print(f"FAIL (expected shape (20, 20), got {tuple(result.shape)})")
        return

    expected = torch.cdist(points, points)
    if not torch.allclose(result, expected, atol=1e-4):
        print(f"FAIL (max diff: {(result - expected).abs().max():.2e})")
        return

    diag = result.diagonal()
    if not torch.allclose(diag, torch.zeros(20), atol=1e-4):
        print(f"FAIL (diagonal should be zero, got max {diag.abs().max():.2e})")
        return

    print("PASS")


def test_einops_flatten(fn):
    print("Testing einops_flatten...", end=" ")
    x = torch.randn(4, 3, 8, 8)
    result = fn(x)
    expected = rearrange(x, "b c h w -> b (c h w)")
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(result.shape)})")
        return
    if not torch.allclose(result, expected, atol=1e-5):
        print(f"FAIL (values don't match, max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")


def test_einops_split_heads(fn):
    print("Testing einops_split_heads...", end=" ")
    x = torch.randn(2, 10, 64)
    result = fn(x, n_heads=8)
    expected = rearrange(x, "b s (h d) -> b h s d", h=8)
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(result.shape)})")
        return
    if not torch.allclose(result, expected, atol=1e-5):
        print(f"FAIL (values don't match, max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")


def test_einops_mean_pool(fn):
    print("Testing einops_mean_pool...", end=" ")
    x = torch.randn(4, 10, 32)
    result = fn(x)
    expected = reduce(x, "b s d -> b d", "mean")
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(result.shape)})")
        return
    if not torch.allclose(result, expected, atol=1e-5):
        print(f"FAIL (values don't match, max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")


def test_quadratic_grad(fn):
    print("Testing quadratic_grad...", end=" ")
    torch.manual_seed(42)
    A = torch.randn(3, 3)
    x = torch.randn(3)
    result = fn(A, x)
    expected = (A + A.T) @ x
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {tuple(expected.shape)}, got {tuple(result.shape)})")
        return
    if not torch.allclose(result, expected, atol=1e-4):
        print(f"FAIL (max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")


def test_network_grad(fn):
    print("Testing network_grad...", end=" ")
    torch.manual_seed(42)
    W = torch.randn(4, 3)
    x = torch.randn(3)
    b = torch.randn(4)
    result = fn(W, x, b)
    if result.shape != (4, 3):
        print(f"FAIL (expected shape (4, 3), got {tuple(result.shape)})")
        return

    W2 = W.clone().requires_grad_(True)
    logits = W2 @ x + b
    torch.softmax(logits, dim=0).sum().backward()
    expected = W2.grad
    if not torch.allclose(result, expected, atol=1e-4):
        print(f"FAIL (max diff: {(result - expected).abs().max():.2e})")
        return
    print("PASS")
