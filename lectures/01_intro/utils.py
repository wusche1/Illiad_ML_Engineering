"""Test functions for Lecture 1."""

import math


def test_tangent(fn, tol=1e-9):
    cases = [
        (0, 0.0),
        (math.pi / 4, 1.0),
        (-math.pi / 4, -1.0),
        (math.pi / 3, math.sqrt(3)),
    ]
    all_passed = True
    for x, expected in cases:
        got = fn(x)
        if got is None:
            print(f"FAIL: tan({x:.4f}) returned None -- not yet implemented?")
            all_passed = False
        elif abs(got - expected) < tol:
            print(f"PASS: tan({x:.4f}) = {got:.6f}")
        else:
            print(f"FAIL: tan({x:.4f}) = {got:.6f}, expected {expected:.6f}")
            all_passed = False
    if all_passed:
        print("\nAll tests passed!")
