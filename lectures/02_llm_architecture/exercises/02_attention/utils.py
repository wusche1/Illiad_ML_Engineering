"""Test functions for Lecture 02: Attention exercise."""
import torch
import torch.nn as nn
import einops
import math


def _reference_single_head_attention(x, W_Q, W_K, W_V, W_O, d_head):
    """Reference implementation of single-head causal self-attention."""
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
    seq_len = x.shape[-2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
    scores.masked_fill_(mask, -1e5)
    pattern = scores.softmax(dim=-1)
    z = pattern @ V
    return z @ W_O


def _reference_multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads, d_head):
    """Reference implementation of multi-head causal self-attention."""
    batch, seq_len, d_model = x.shape

    Q = einops.einsum(x, W_Q, "batch seq d_model, head d_model d_head -> batch seq head d_head")
    K = einops.einsum(x, W_K, "batch seq d_model, head d_model d_head -> batch seq head d_head")
    V = einops.einsum(x, W_V, "batch seq d_model, head d_model d_head -> batch seq head d_head")

    scores = einops.einsum(Q, K,
        "batch seq_q head d_head, batch seq_k head d_head -> batch head seq_q seq_k")
    scores = scores / math.sqrt(d_head)

    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
    scores.masked_fill_(mask, -1e5)
    pattern = scores.softmax(dim=-1)

    z = einops.einsum(pattern, V,
        "batch head seq_q seq_k, batch seq_k head d_head -> batch seq_q head d_head")

    out = einops.einsum(z, W_O,
        "batch seq_q head d_head, head d_head d_model -> batch seq_q d_model")
    return out


def test_causal_mask(mask_fn):
    print("Testing causal mask...", end=" ")
    scores = torch.zeros(1, 1, 4, 4)
    masked = mask_fn(scores)
    expected_mask = torch.tensor([
        [0., -1e5, -1e5, -1e5],
        [0., 0., -1e5, -1e5],
        [0., 0., 0., -1e5],
        [0., 0., 0., 0.],
    ])
    if torch.allclose(masked[0, 0], expected_mask, atol=1):
        print("PASS")
    else:
        print(f"FAIL\nExpected:\n{expected_mask}\nGot:\n{masked[0, 0]}")


def test_attention_pattern(pattern_fn):
    print("Testing attention pattern...", end=" ")
    torch.manual_seed(42)
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    d_head = 8

    pattern = pattern_fn(Q, K, d_head)

    if pattern.shape != (1, 4, 4):
        print(f"FAIL (wrong shape: got {pattern.shape}, expected (1, 4, 4))")
        return

    # Check rows sum to 1
    row_sums = pattern.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        print(f"FAIL (rows don't sum to 1: {row_sums})")
        return

    # Check causal: upper triangle should be ~0
    upper = pattern[0].triu(diagonal=1)
    if upper.abs().max() > 1e-4:
        print(f"FAIL (non-zero values in upper triangle: max={upper.abs().max():.2e})")
        return

    print("PASS")


def test_single_head_attention(AttentionClass, d_model=32, d_head=16):
    print("Testing single-head attention...", end=" ")
    torch.manual_seed(42)
    attn = AttentionClass(d_model=d_model, d_head=d_head)

    W_Q = attn.W_Q.data.clone()
    W_K = attn.W_K.data.clone()
    W_V = attn.W_V.data.clone()
    W_O = attn.W_O.data.clone()

    x = torch.randn(2, 6, d_model)
    got = attn(x)
    expected = _reference_single_head_attention(x, W_Q, W_K, W_V, W_O, d_head)

    if got.shape != expected.shape:
        print(f"FAIL (wrong shape: got {got.shape}, expected {expected.shape})")
        return
    if torch.allclose(got, expected, atol=1e-4):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(got - expected).abs().max():.2e})")


def test_multi_head_attention(MHAClass, d_model=64, n_heads=4, d_head=16):
    print("Testing multi-head attention...", end=" ")
    torch.manual_seed(42)
    attn = MHAClass(d_model=d_model, n_heads=n_heads, d_head=d_head)

    W_Q = attn.W_Q.data.clone()
    W_K = attn.W_K.data.clone()
    W_V = attn.W_V.data.clone()
    W_O = attn.W_O.data.clone()

    x = torch.randn(2, 8, d_model)
    got = attn(x)
    expected = _reference_multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads, d_head)

    if got.shape != expected.shape:
        print(f"FAIL (wrong shape: got {got.shape}, expected {expected.shape})")
        return
    if torch.allclose(got, expected, atol=1e-4):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(got - expected).abs().max():.2e})")
