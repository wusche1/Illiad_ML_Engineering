"""Solutions for PyTorch basics exercises."""

SOLUTIONS = {
    "pairwise_distances": """\
def pairwise_distances(points):
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    return diff.norm(dim=-1)
""",
    "einops_row": """\
result = rearrange(arr, 'b c h w -> c h (b w)')
""",
    "einops_grid": """\
result = rearrange(arr, '(r cols) c h w -> c (r h) (cols w)', r=2)
""",
    "einops_stretch": """\
result = repeat(arr[0], 'c h w -> c (h 2) w')
""",
    "einops_channels": """\
result = rearrange(arr[0], 'c h w -> h (c w)')
""",
    "einops_maxpool": """\
result = reduce(arr, '(r cols) c (h h2) (w w2) -> c (r h) (cols w)', 'max', r=2, h2=2, w2=2)
""",
    "quadratic_grad": """\
def quadratic_grad(A, x):
    x = x.clone().requires_grad_(True)
    f = x @ A @ x
    f.backward()
    return x.grad
""",
    "network_grad": """\
def network_grad(W, x, b):
    W = W.clone().requires_grad_(True)
    logits = W @ x + b
    f = torch.softmax(logits, dim=0).sum()
    f.backward()
    return W.grad
""",
}
