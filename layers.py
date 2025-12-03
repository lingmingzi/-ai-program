import numpy as np


class Conv2D:
    def __init__(self, in_ch, out_ch, k, stride=1, pad=0, optimizer=None, key=None):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.stride = stride
        self.pad = pad
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = np.random.randn(out_ch, in_ch, k, k).astype(np.float32) * scale
        self.b = np.zeros((out_ch, 1), dtype=np.float32)
        self.optimizer = optimizer
        self.key = key if key is not None else f"conv_{id(self)}"

    def forward(self, x):
        # x: (N, C, H, W)
        self.x = x
        N, C, H, W = x.shape
        OC, _, K, _ = self.W.shape
        OH, OW = H - K + 1, W - K + 1
        out = np.zeros((N, OC, OH, OW), dtype=np.float32)
        for n in range(N):
            for oc in range(OC):
                for i in range(OH):
                    for j in range(OW):
                        out[n, oc, i, j] = np.sum(x[n, :, i:i+K, j:j+K] * self.W[oc]) + self.b[oc]
        return out

    def backward(self, grad_out):
        x = self.x
        N, C, H, W = x.shape
        OC, _, K, _ = self.W.shape
        OH, OW = grad_out.shape[2], grad_out.shape[3]

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(x)

        for n in range(N):
            for oc in range(OC):
                db[oc] += np.sum(grad_out[n, oc])
                for i in range(OH):
                    for j in range(OW):
                        region = x[n, :, i:i+K, j:j+K]
                        dW[oc] += grad_out[n, oc, i, j] * region
                        dx[n, :, i:i+K, j:j+K] += grad_out[n, oc, i, j] * self.W[oc]

        # update params via optimizer if given
        if self.optimizer is not None:
            self.W, self.b = self.optimizer.update(self.W, dW, self.b, db, key=self.key)
        else:
            self.W -= 1e-3 * dW
            self.b -= 1e-3 * db

        return dx


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class MaxPool2x2:
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h, out_w = H // 2, W // 2
        out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
        self.mask = np.zeros_like(x, dtype=bool)
        for i in range(out_h):
            for j in range(out_w):
                block = x[:, :, 2*i:2*i+2, 2*j:2*j+2]
                maxv = np.max(block, axis=(2, 3), keepdims=True)
                out[:, :, i, j] = maxv.squeeze()
                self.mask[:, :, 2*i:2*i+2, 2*j:2*j+2] = (block == maxv)
        return out

    def backward(self, grad):
        N, C, H, W = self.x.shape
        dx = np.zeros_like(self.x)
        for i in range(grad.shape[2]):
            for j in range(grad.shape[3]):
                dx[:, :, 2*i:2*i+2, 2*j:2*j+2] += self.mask[:, :, 2*i:2*i+2, 2*j:2*j+2] * grad[:, :, i, j][:, :, None, None]
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class Dense:
    def __init__(self, in_dim, out_dim, optimizer=None, key=None):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.optimizer = optimizer
        self.key = key if key is not None else f"dense_{id(self)}"

    def forward(self, x):
        self.x = x  # (N, in_dim)
        return x.dot(self.W) + self.b

    def backward(self, grad):
        dW = self.x.T.dot(grad)
        db = np.sum(grad, axis=0)
        dx = grad.dot(self.W.T)
        if self.optimizer is not None:
            self.W, self.b = self.optimizer.update(self.W, dW, self.b, db, key=self.key)
        else:
            self.W -= 1e-3 * dW
            self.b -= 1e-3 * db
        return dx


class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, optimizer=None, key=None, for_conv=True):
        self.for_conv = for_conv
        self.eps = eps
        self.momentum = momentum
        if self.for_conv:
            # gamma/beta shaped (1,C,1,1) to broadcast
            self.gamma = np.ones((1, num_features, 1, 1), dtype=np.float32)
            self.beta = np.zeros((1, num_features, 1, 1), dtype=np.float32)
            self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
            self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)
        else:
            self.gamma = np.ones((1, num_features), dtype=np.float32)
            self.beta = np.zeros((1, num_features), dtype=np.float32)
            self.running_mean = np.zeros((1, num_features), dtype=np.float32)
            self.running_var = np.ones((1, num_features), dtype=np.float32)
        self.optimizer = optimizer
        self.key = key if key is not None else f"bn_{id(self)}"

    def forward(self, x, training=True):
        self.x_shape = x.shape
        self.x = x
        self.training = training
        
        if self.for_conv:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            if training:
                self.mean = mean
                self.var = var
                self.x_hat = (x - mean) / np.sqrt(var + self.eps)
                out = self.gamma * self.x_hat + self.beta
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
                return out
            else:
                self.mean = self.running_mean
                self.var = self.running_var
                self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
                return self.gamma * self.x_hat + self.beta
        else:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            if training:
                self.mean = mean
                self.var = var
                self.x_hat = (x - mean) / np.sqrt(var + self.eps)
                out = self.gamma * self.x_hat + self.beta
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
                return out
            else:
                self.mean = self.running_mean
                self.var = self.running_var
                self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
                return self.gamma * self.x_hat + self.beta

    def backward(self, grad_out):
        if self.for_conv:
            x = self.x_hat
            m = np.prod(self.x_shape) / self.x_shape[1]  # N*H*W
            dx_hat = grad_out * self.gamma
            dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps)**(-3/2), axis=(0, 2, 3), keepdims=True)
            dmean = np.sum(dx_hat * (-1.0 / np.sqrt(self.var + self.eps)), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2.0 * (self.x - self.mean), axis=(0, 2, 3), keepdims=True) / m
            dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0 * (self.x - self.mean) / m + dmean / m

            # grads for gamma/beta
            dgamma = np.sum(grad_out * self.x_hat, axis=(0, 2, 3), keepdims=True)
            dbeta = np.sum(grad_out, axis=(0, 2, 3), keepdims=True)

            # update gamma/beta
            if self.optimizer is not None:
                g_key = self.key + "_g"
                b_key = self.key + "_b"
                self.gamma, _ = self.optimizer.update(self.gamma, dgamma, None, None, key=g_key)
                self.beta, _ = self.optimizer.update(self.beta, dbeta, None, None, key=b_key)
            else:
                self.gamma -= 1e-3 * dgamma
                self.beta -= 1e-3 * dbeta

            return dx
        else:
            x = self.x_hat
            m = self.x_shape[0]
            dx_hat = grad_out * self.gamma
            dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps)**(-3/2), axis=0, keepdims=True)
            dmean = np.sum(dx_hat * (-1.0 / np.sqrt(self.var + self.eps)), axis=0, keepdims=True) + dvar * np.sum(-2.0 * (self.x - self.mean), axis=0, keepdims=True) / m
            dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0 * (self.x - self.mean) / m + dmean / m

            dgamma = np.sum(grad_out * self.x_hat, axis=0, keepdims=True)
            dbeta = np.sum(grad_out, axis=0, keepdims=True)
            if self.optimizer is not None:
                g_key = self.key + "_g"
                b_key = self.key + "_b"
                self.gamma, _ = self.optimizer.update(self.gamma, dgamma, None, None, key=g_key)
                self.beta, _ = self.optimizer.update(self.beta, dbeta, None, None, key=b_key)
            else:
                self.gamma -= 1e-3 * dgamma
                self.beta -= 1e-3 * dbeta
            return dx


class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            return x * self.mask
        else:
            return x

    def backward(self, grad):
        return grad * self.mask
