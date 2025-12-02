import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, lr=1e-2):
        self.lr = lr

    def update(self, W, dW, b=None, db=None, key=None):
        W = W - self.lr * dW
        if b is not None:
            b = b - self.lr * db
        return W, b


class Momentum:
    """Momentum optimizer with velocity tracking."""
    def __init__(self, lr=1e-2, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, W, dW, b=None, db=None, key=None):
        if key is None:
            key = id(W)
        if key not in self.vW:
            self.vW[key] = np.zeros_like(W)
            if b is not None:
                self.vb[key] = np.zeros_like(b)
        self.vW[key] = self.beta * self.vW[key] + (1 - self.beta) * dW
        W = W - self.lr * self.vW[key]
        if b is not None:
            self.vb[key] = self.beta * self.vb[key] + (1 - self.beta) * db
            b = b - self.lr * self.vb[key]
        return W, b


class RMSProp:
    """RMSProp optimizer with adaptive learning rate."""
    def __init__(self, lr=1e-3, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.sW = {}
        self.sb = {}

    def update(self, W, dW, b=None, db=None, key=None):
        if key is None:
            key = id(W)
        if key not in self.sW:
            self.sW[key] = np.zeros_like(W)
            if b is not None:
                self.sb[key] = np.zeros_like(b)
        self.sW[key] = self.rho * self.sW[key] + (1 - self.rho) * (dW**2)
        W = W - self.lr * dW / (np.sqrt(self.sW[key]) + self.eps)
        if b is not None:
            self.sb[key] = self.rho * self.sb[key] + (1 - self.rho) * (db**2)
            b = b - self.lr * db / (np.sqrt(self.sb[key]) + self.eps)
        return W, b


class Adam:
    """Adam optimizer with momentum and adaptive learning rate."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}
        self.t = {}

    def update(self, W, dW, b=None, db=None, key=None):
        if key is None:
            key = id(W)
        if key not in self.mW:
            self.mW[key] = np.zeros_like(W)
            self.vW[key] = np.zeros_like(W)
            self.t[key] = 0
            if b is not None:
                self.mb[key] = np.zeros_like(b)
                self.vb[key] = np.zeros_like(b)
        self.t[key] += 1
        t = self.t[key]
        self.mW[key] = self.beta1 * self.mW[key] + (1 - self.beta1) * dW
        self.vW[key] = self.beta2 * self.vW[key] + (1 - self.beta2) * (dW**2)
        m_hat = self.mW[key] / (1 - self.beta1**t)
        v_hat = self.vW[key] / (1 - self.beta2**t)
        W = W - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if b is not None:
            self.mb[key] = self.beta1 * self.mb[key] + (1 - self.beta1) * db
            self.vb[key] = self.beta2 * self.vb[key] + (1 - self.beta2) * (db**2)
            mb_hat = self.mb[key] / (1 - self.beta1**t)
            vb_hat = self.vb[key] / (1 - self.beta2**t)
            b = b - self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        return W, b
