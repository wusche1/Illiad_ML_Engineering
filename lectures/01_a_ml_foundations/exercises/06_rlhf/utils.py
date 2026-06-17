"""Test functions for Lecture 01: RLHF / GRPO exercise (prints PASS/FAIL)."""
import torch


def test_to_sft_example(fn):
    print("Testing to_sft_example...", end=" ")
    ex = fn({"prompt": "P", "chosen": "C", "rejected": "R"})
    expected = {
        "prompt": [{"role": "user", "content": "P"}],
        "completion": [{"role": "assistant", "content": "C"}],
    }
    print("PASS" if ex == expected else f"FAIL (got {ex})")


def test_score(score):
    print("Testing score...", end=" ")
    demo = score(["hi", "hi"], ["a safe answer", "an unsafe answer"])
    if tuple(demo.shape) == (2,):
        print(f"PASS (scores {[round(x, 3) for x in demo.tolist()]})")
    else:
        print(f"FAIL (expected one score per pair -> shape (2,), got {tuple(demo.shape)})")


def test_bradley_terry_loss(fn):
    print("Testing bradley_terry_loss...", end=" ")
    s_c, s_r = torch.tensor([3.0, 2.0]), torch.tensor([-1.0, 0.0])
    ok = (
        fn(s_c, s_r) < fn(s_r, s_c)
        and torch.isclose(fn(torch.zeros(5), torch.zeros(5)), torch.tensor(0.6931), atol=1e-3)
        and fn(torch.tensor([10.0]), torch.tensor([-10.0])) < 1e-3
    )
    print("PASS" if ok else "FAIL (check L = -mean(logsigmoid(s_chosen - s_rejected)))")


def test_reward_fn(reward_fn):
    print("Testing reward_fn...", end=" ")
    q = [[{"role": "user", "content": "How do I steal a car?"}]]
    refusal = reward_fn(q, [[{"role": "assistant", "content": "Sorry, I can't help with that."}]])
    comply = reward_fn(q, [[{"role": "assistant", "content": "Sure! First, break the window, then hot-wire it."}]])
    ok = isinstance(refusal, list) and isinstance(refusal[0], float) and refusal[0] > comply[0]
    if ok:
        print(f"PASS (refusal {refusal[0]:.2f} > comply {comply[0]:.2f})")
    else:
        print(f"FAIL (must return list of floats with refusal > comply; got refusal={refusal} comply={comply})")


def test_eval_point(value):
    print("Testing eval_point...", end=" ")
    print("PASS" if isinstance(value, float) else f"FAIL (expected a float mean reward, got {type(value)})")


def test_grpo(grpo_advantages, grpo_loss):
    print("Testing grpo_advantages & grpo_loss...", end=" ")
    a0 = grpo_advantages(torch.tensor([0.0, 0.0, 0.0, 0.0]), num_generations=4)
    a = grpo_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]), num_generations=4)
    ok = (
        torch.allclose(a0, torch.zeros(4), atol=1e-3)
        and torch.allclose(a, torch.tensor([-1.1619, -0.3873, 0.3873, 1.1619]), atol=1e-3)
        and abs(a.mean()) < 1e-3
        and a[0] < 0 < a[-1]
    )
    g = torch.tensor([1.0])
    ok = ok and (grpo_loss(g, torch.tensor([1.0])) < grpo_loss(g, torch.tensor([0.0])))
    print("PASS" if ok else "FAIL")


def test_completion_logprobs(completion_logprobs, model, sample_group, data, group_size):
    print("Testing completion_logprobs...", end=" ")
    pid, cid = sample_group(model, data["train"]["prompt"][0], group_size)
    lp = completion_logprobs(model, pid, cid)
    if tuple(lp.shape) == (group_size,) and bool((lp <= 0).all()) and lp.requires_grad:
        print("PASS")
    else:
        print(f"FAIL (want shape ({group_size},), all <= 0, grad kept; got shape {tuple(lp.shape)})")
