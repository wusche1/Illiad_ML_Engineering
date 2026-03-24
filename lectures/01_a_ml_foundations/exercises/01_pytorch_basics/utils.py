"""Test functions for PyTorch basics exercises."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat


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


# ── Digit image generation ──

def make_digit_images():
    """Generate 6 colorful digit images (shape: 6, 3, 100, 100)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    images = []
    for i, color in enumerate(colors):
        fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.45, str(i), fontsize=72, ha='center', va='center',
                fontweight='bold', color=color, family='monospace')
        ax.set_facecolor('black')
        ax.axis('off')
        fig.patch.set_facecolor('black')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = torch.tensor(buf[:, :, :3], dtype=torch.float32).permute(2, 0, 1) / 255.0
        images.append(img)
        plt.close(fig)
    return torch.stack(images)


def show(img, title=None):
    """Display a tensor as an image. Handles (C,H,W) RGB and (H,W) grayscale."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    fig, ax = plt.subplots(1, 1, figsize=(max(3, img.shape[1] / 40), max(2, img.shape[0] / 40)))
    if img.ndim == 2:
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(img.clamp(0, 1).numpy())
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ── Expected outputs for einops exercises (hidden from notebook) ──

def _expected(arr, exercise_id):
    targets = {
        "einops_row": lambda a: rearrange(a, 'b c h w -> c h (b w)'),
        "einops_grid": lambda a: rearrange(a, '(r cols) c h w -> c (r h) (cols w)', r=2),
        "einops_stretch": lambda a: repeat(a[0], 'c h w -> c (h 2) w'),
        "einops_channels": lambda a: rearrange(a[0], 'c h w -> h (c w)'),
        "einops_maxpool": lambda a: reduce(a, '(r cols) c (h h2) (w w2) -> c (r h) (cols w)', 'max', r=2, h2=2, w2=2),
    }
    return targets[exercise_id](arr)


def show_target(arr, exercise_id):
    """Show the target output for an einops exercise."""
    result = _expected(arr, exercise_id)
    show(result, title=f"Target (shape: {list(result.shape)})")


# ── Einops test functions ──

def _test_einops(name, result, arr, exercise_id):
    print(f"Testing {name}...", end=" ")
    expected = _expected(arr, exercise_id)
    if result is None:
        print("FAIL (returned None)")
        return
    if not isinstance(result, torch.Tensor):
        print(f"FAIL (expected tensor, got {type(result).__name__})")
        return
    if result.shape != expected.shape:
        print(f"FAIL (expected shape {list(expected.shape)}, got {list(result.shape)})")
        return
    if not torch.allclose(result.float(), expected.float(), atol=1e-4):
        print(f"FAIL (values don't match, max diff: {(result.float() - expected.float()).abs().max():.2e})")
        return
    print("PASS")
    show(result, title=f"{name} — correct!")


def test_einops_row(result, arr):
    _test_einops("einops_row", result, arr, "einops_row")


def test_einops_grid(result, arr):
    _test_einops("einops_grid", result, arr, "einops_grid")


def test_einops_stretch(result, arr):
    _test_einops("einops_stretch", result, arr, "einops_stretch")


def test_einops_channels(result, arr):
    _test_einops("einops_channels", result, arr, "einops_channels")


def test_einops_maxpool(result, arr):
    _test_einops("einops_maxpool", result, arr, "einops_maxpool")


# ── Autograd tests ──

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
