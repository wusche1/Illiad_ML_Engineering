"""Test functions for Lecture 02: MLP exercise."""
import torch
import torch.nn as nn
import einops


def gelu_new(x):
    """GPT-2's GELU approximation."""
    return 0.5 * x * (1.0 + torch.tanh(
        (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3.0))
    ))


def _make_reference_mlp(d_model, d_mlp):
    """Build a reference MLP using nn.Linear for testing."""
    W_in = torch.randn(d_model, d_mlp) * 0.02
    b_in = torch.randn(d_mlp) * 0.02
    W_out = torch.randn(d_mlp, d_model) * 0.02
    b_out = torch.randn(d_model) * 0.02
    return W_in, b_in, W_out, b_out


def _reference_mlp_forward(x, W_in, b_in, W_out, b_out):
    pre = x @ W_in + b_in
    post = gelu_new(pre)
    return post @ W_out + b_out


def test_gelu(student_gelu):
    print("Testing GELU...", end=" ")
    x = torch.linspace(-3, 3, 100)
    expected = gelu_new(x)
    got = student_gelu(x)
    if torch.allclose(got, expected, atol=1e-5):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(got - expected).abs().max():.2e})")


def test_mlp(MLPClass, d_model=64, d_mlp=256):
    print("Testing MLP...", end=" ")
    torch.manual_seed(42)
    mlp = MLPClass(d_model=d_model, d_mlp=d_mlp)

    # Set known weights
    W_in, b_in, W_out, b_out = _make_reference_mlp(d_model, d_mlp)
    with torch.no_grad():
        mlp.W_in.copy_(W_in)
        mlp.b_in.copy_(b_in)
        mlp.W_out.copy_(W_out)
        mlp.b_out.copy_(b_out)

    x = torch.randn(2, 10, d_model)
    expected = _reference_mlp_forward(x, W_in, b_in, W_out, b_out)
    got = mlp(x)

    if got.shape != expected.shape:
        print(f"FAIL (wrong shape: got {got.shape}, expected {expected.shape})")
        return
    if torch.allclose(got, expected, atol=1e-4):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(got - expected).abs().max():.2e})")


def test_mlp_shapes(MLPClass, d_model=64, d_mlp=256):
    print("Testing MLP shapes...", end=" ")
    mlp = MLPClass(d_model=d_model, d_mlp=d_mlp)
    x = torch.randn(2, 10, d_model)
    out = mlp(x)
    if out.shape == (2, 10, d_model):
        print("PASS")
    else:
        print(f"FAIL (expected (2, 10, {d_model}), got {out.shape})")


def test_residual_mlp(ResidualMLPClass, d_model=64, d_mlp=256, n_layers=3):
    print("Testing Residual MLP stack...", end=" ")
    torch.manual_seed(0)
    model = ResidualMLPClass(d_model=d_model, d_mlp=d_mlp, n_layers=n_layers)
    x = torch.randn(2, 10, d_model)
    out = model(x)

    if out.shape != x.shape:
        print(f"FAIL (wrong shape: got {out.shape}, expected {x.shape})")
        return

    # Check it's not just identity (residual connections + random weights should change values)
    if torch.allclose(out, x, atol=1e-3):
        print("FAIL (output equals input; are you applying the MLP layers?)")
        return

    # Check gradient flows through all layers
    x2 = torch.randn(2, 10, d_model, requires_grad=True)
    out2 = model(x2)
    out2.sum().backward()
    if x2.grad is not None and x2.grad.abs().sum() > 0:
        print("PASS")
    else:
        print("FAIL (no gradient flows back to input)")
