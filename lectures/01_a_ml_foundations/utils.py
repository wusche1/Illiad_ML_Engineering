"""Test functions for Lecture 01a exercises."""
import torch


def _run_optimizer_test(name, student_opt, ref_opt, student_params, ref_params, num_steps=5):
    all_passed = True
    for step in range(1, num_steps + 1):
        loss_s = sum((p ** 2).sum() for p in student_params)
        loss_r = sum((p ** 2).sum() for p in ref_params)
        loss_s.backward()
        loss_r.backward()
        student_opt.step()
        ref_opt.step()
        student_opt.zero_grad()
        ref_opt.zero_grad()
        match = all(
            torch.allclose(ps.data, pr.data, atol=1e-6)
            for ps, pr in zip(student_params, ref_params)
        )
        if match:
            print(f"  Step {step}: PASS")
        else:
            max_diff = max(
                (ps.data - pr.data).abs().max().item()
                for ps, pr in zip(student_params, ref_params)
            )
            print(f"  Step {step}: FAIL (max diff: {max_diff:.2e})")
            all_passed = False
    if all_passed:
        print(f"\n{name}: All tests passed!")
    return all_passed


def test_vanilla_sgd(SGDClass):
    print("Testing VanillaSGD...")
    torch.manual_seed(42)
    p1 = [torch.randn(4, 3, requires_grad=True), torch.randn(5, requires_grad=True)]
    p2 = [p.data.clone().requires_grad_(True) for p in p1]
    _run_optimizer_test(
        "VanillaSGD",
        SGDClass(p1, lr=0.1),
        torch.optim.SGD(p2, lr=0.1),
        p1, p2,
    )


def test_sgd_momentum(SGDMomentumClass):
    print("Testing SGDMomentum...")
    torch.manual_seed(42)
    p1 = [torch.randn(4, 3, requires_grad=True), torch.randn(5, requires_grad=True)]
    p2 = [p.data.clone().requires_grad_(True) for p in p1]
    _run_optimizer_test(
        "SGDMomentum",
        SGDMomentumClass(p1, lr=0.01, momentum=0.9),
        torch.optim.SGD(p2, lr=0.01, momentum=0.9),
        p1, p2,
    )


def test_adam(AdamClass):
    print("Testing Adam...")
    torch.manual_seed(42)
    p1 = [torch.randn(4, 3, requires_grad=True), torch.randn(5, requires_grad=True)]
    p2 = [p.data.clone().requires_grad_(True) for p in p1]
    _run_optimizer_test(
        "Adam",
        AdamClass(p1, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
        torch.optim.Adam(p2, lr=0.001, betas=(0.9, 0.999), eps=1e-8),
        p1, p2,
    )
