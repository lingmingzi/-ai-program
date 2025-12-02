import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy_loss_and_grad(logits, y):
    N = logits.shape[0]
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-8))
    grad = probs.copy()
    grad[np.arange(N), y] -= 1
    grad /= N
    return loss, grad


def accuracy(probs, y):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)
