"""Test functions for architecture exercises."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_residual_block(BlockClass):
    print("Testing ResidualBlock...", end=" ")
    torch.manual_seed(42)
    block = BlockClass(dim=16)
    x = torch.randn(4, 16)
    out = block(x)

    if out.shape != x.shape:
        print(f"FAIL (expected shape {tuple(x.shape)}, got {tuple(out.shape)})")
        return

    with torch.no_grad():
        cosine_sim = F.cosine_similarity(x.flatten(), out.flatten(), dim=0).item()
    if cosine_sim < 0.5:
        print(f"FAIL (no skip connection detected, cosine_sim={cosine_sim:.2f})")
        return

    if torch.allclose(out, x, atol=1e-4):
        print("FAIL (output identical to input, F(x) seems to be zero)")
        return

    print("PASS")


def test_tiny_cnn(CNNClass):
    print("Testing TinyCNN...", end=" ")
    torch.manual_seed(42)
    model = CNNClass()
    x = torch.randn(8, 1, 8, 8)
    out = model(x)

    if out.shape != (8, 2):
        print(f"FAIL (expected output shape (8, 2), got {tuple(out.shape)})")
        return

    torch.manual_seed(42)
    model = CNNClass()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train = torch.randn(32, 1, 8, 8)
    y_train = torch.randint(0, 2, (32,))
    loss_start = F.cross_entropy(model(x_train), y_train).item()
    for _ in range(50):
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss_end = loss.item()
    if loss_end >= loss_start:
        print(f"FAIL (loss did not decrease: {loss_start:.3f} -> {loss_end:.3f})")
        return

    print("PASS")
