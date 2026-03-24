"""Test and visualization functions for Lecture 01a exercises."""
import torch
import numpy as np
import matplotlib.pyplot as plt


# ── Rosenbrock's banana function ──

def rosenbrocks_banana(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def optimize(fn, parameters, optimizer, n_steps):
    trajectory = []
    for _ in range(n_steps):
        trajectory.append(parameters.detach().clone())
        loss = fn(*parameters)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return torch.stack(trajectory).float()


def plot_banana(trajectories, xmin=-2, xmax=2, ymin=-1, ymax=3, n=50):
    x = torch.linspace(xmin, xmax, n)
    y = torch.linspace(ymin, ymax, n)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    zz = rosenbrocks_banana(xx, yy)

    fig = plt.figure(figsize=(13, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx.numpy(), yy.numpy(), zz.numpy(), cmap='viridis', alpha=0.5)
    for name, traj in trajectories.items():
        z = rosenbrocks_banana(traj[:, 0], traj[:, 1])
        ax1.plot(traj[:, 0], traj[:, 1], z, label=name, linewidth=3)
    ax1.scatter(1, 1, float(rosenbrocks_banana(torch.tensor(1.), torch.tensor(1.))),
                color='red', s=60, zorder=5, label='Minimum')
    ax1.set(xlabel='x', ylabel='y', zlabel='Loss')
    ax1.legend(fontsize=8)

    # Contour
    ax2 = fig.add_subplot(122)
    levels = np.logspace(np.log10(float(zz.min())), np.log10(float(zz.max())), 15)
    ax2.contour(x.numpy(), y.numpy(), zz.numpy(), levels=levels, cmap='viridis')
    for name, traj in trajectories.items():
        ax2.plot(traj[:, 0], traj[:, 1], '.-', label=name, linewidth=2, markersize=3)
    ax2.scatter(1, 1, color='red', s=60, zorder=5, label='Minimum')
    ax2.set(xlabel='x', ylabel='y')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── Optimizer tests (against PyTorch reference on the banana) ──

def test_sgd(SGDClass):
    print("Testing SGD...", end=" ")
    p1 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj1 = optimize(rosenbrocks_banana, p1, SGDClass(p1, lr=0.001), 100)
    p2 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj2 = optimize(rosenbrocks_banana, p2, torch.optim.SGD([p2], lr=0.001), 100)
    if torch.allclose(traj1, traj2, atol=1e-3):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(traj1 - traj2).abs().max():.2e})")


def test_momentum(MomentumClass):
    print("Testing Momentum...", end=" ")
    p1 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj1 = optimize(rosenbrocks_banana, p1, MomentumClass(p1, lr=0.001, momentum=0.9), 100)
    p2 = torch.tensor([-1.0, 2.0], requires_grad=True)
    # Student: v = β*v + (1-β)*g, θ -= lr*v. PyTorch: buf = β*buf + g, θ -= lr*buf.
    # Equivalent when lr_pytorch = lr_student * (1-β)
    traj2 = optimize(rosenbrocks_banana, p2,
                     torch.optim.SGD([p2], lr=0.001 * (1 - 0.9), momentum=0.9), 100)
    if torch.allclose(traj1, traj2, atol=1e-3):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(traj1 - traj2).abs().max():.2e})")


def test_rmsprop(RMSPropClass):
    print("Testing RMSProp...", end=" ")
    p1 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj1 = optimize(rosenbrocks_banana, p1,
                     RMSPropClass(p1, lr=0.001, beta=0.9, eps=1e-8), 100)
    p2 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj2 = optimize(rosenbrocks_banana, p2,
                     torch.optim.RMSprop([p2], lr=0.001, alpha=0.9, eps=1e-8), 100)
    if torch.allclose(traj1, traj2, atol=1e-3):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(traj1 - traj2).abs().max():.2e})")


def test_adam(AdamClass):
    print("Testing Adam...", end=" ")
    p1 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj1 = optimize(rosenbrocks_banana, p1,
                     AdamClass(p1, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8), 100)
    p2 = torch.tensor([-1.0, 2.0], requires_grad=True)
    traj2 = optimize(rosenbrocks_banana, p2,
                     torch.optim.Adam([p2], lr=0.001, betas=(0.9, 0.999), eps=1e-8), 100)
    if torch.allclose(traj1, traj2, atol=1e-3):
        print("PASS")
    else:
        print(f"FAIL (max diff: {(traj1 - traj2).abs().max():.2e})")
