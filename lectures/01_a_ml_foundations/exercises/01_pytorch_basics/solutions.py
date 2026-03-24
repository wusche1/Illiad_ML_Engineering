"""Solutions for PyTorch basics exercises."""

SOLUTIONS = {
    "pairwise_distances": """\
def pairwise_distances(points):
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    return diff.norm(dim=-1)
""",
    "einops_flatten": """\
def einops_flatten(x):
    return rearrange(x, 'b c h w -> b (c h w)')
""",
    "einops_split_heads": """\
def einops_split_heads(x, n_heads):
    return rearrange(x, 'b s (h d) -> b h s d', h=n_heads)
""",
    "einops_mean_pool": """\
def einops_mean_pool(x):
    return reduce(x, 'b s d -> b d', 'mean')
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
