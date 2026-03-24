"""Single source of truth for optimizer exercise solutions."""

SOLUTIONS = {
    "sgd": """\
class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        with torch.no_grad():
            self.parameters -= self.lr * self.parameters.grad

    def zero_grad(self):
        self.parameters.grad.zero_()
""",
    "momentum": """\
class Momentum:
    def __init__(self, parameters, lr=0.001, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.v = torch.zeros_like(parameters)

    def step(self):
        with torch.no_grad():
            self.v = self.momentum * self.v + (1 - self.momentum) * self.parameters.grad
            self.parameters -= self.lr * self.v

    def zero_grad(self):
        self.parameters.grad.zero_()
""",
    "rmsprop": """\
class RMSProp:
    def __init__(self, parameters, lr=0.001, beta=0.9, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.r = torch.zeros_like(parameters)

    def step(self):
        with torch.no_grad():
            self.r = self.beta * self.r + (1 - self.beta) * self.parameters.grad ** 2
            self.parameters -= self.lr / (self.r.sqrt() + self.eps) * self.parameters.grad

    def zero_grad(self):
        self.parameters.grad.zero_()
""",
    "adam": """\
class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = torch.zeros_like(parameters)
        self.v = torch.zeros_like(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            g = self.parameters.grad
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            self.parameters -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        self.parameters.grad.zero_()
""",
}
