import math

import torch


class Optimizer:
    def __init__(self, params):
        self.params = params

    def update(self, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def update(self, grads):
        for [(k1, param), (k2, grad)] in zip(self.params, grads):
            param += self.lr * grad


class Momentum(Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(v) for k, v in params]

    def update(self, grads):
        for [(k1, param), (k2, grad), v] in zip(self.params, grads, self.v):
            assert k1 == k2
            v.copy_(self.momentum * v + (1 - self.momentum) * grad)
            step = self.lr * v
            param.copy_(param + step)


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(v) for k, v in params]
        self.v = [torch.zeros_like(v) for k, v in params]
        self.t = 0

    def update(self, grads):
        self.t += 1
        for [(k1, param), (k2, grad), m, v] in zip(self.params, grads, self.m, self.v):
            assert k1 == k2
            a = (
                self.lr
                * math.sqrt(1 - self.beta2**self.t)
                / (1 - self.beta1**self.t)
            )
            m.copy_(self.beta1 * m + (1 - self.beta1) * grad)
            v.copy_(self.beta2 * v + (1 - self.beta2) * (grad * grad))
            step = a * m / (torch.sqrt(v) + self.epsilon)
            param.copy_(param + step)
