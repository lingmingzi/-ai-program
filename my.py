# train_cnn_from_scratch.py
import os
import urllib.request
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

DATASET_DIR = 'Dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_ARCHIVE = os.path.join(DATASET_DIR, "cifar-10-python.tar.gz")
CIFAR_DIR = os.path.join(DATASET_DIR, "cifar-10-batches-py")

def download_and_extract_cifar():
    if os.path.exists(CIFAR_DIR):
        print("CIFAR-10 found, skipping download.")
        return
    print("Downloading CIFAR-10 (~170MB)...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r进度: {percent:.1f}% ({downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB)", end='')
    
    urllib.request.urlretrieve(CIFAR_URL, CIFAR_ARCHIVE, reporthook=download_progress)
    print("\n提取中...")
    with tarfile.open(CIFAR_ARCHIVE, "r:gz") as f:
        f.extractall(DATASET_DIR)
    print("完成!")
    if os.path.exists(CIFAR_ARCHIVE):
        os.remove(CIFAR_ARCHIVE)

def load_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        X = d['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d['labels'], dtype=np.int64)
        return X, y

def load_cifar10():
    download_and_extract_cifar()
    xs, ys = [], []
    for i in range(1,6):
        X, y = load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
        xs.append(X); ys.append(y)
    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    X_test, y_test = load_batch(os.path.join(CIFAR_DIR, "test_batch"))
    return X_train, y_train, X_test, y_test

def data_augmentation(X):
    """简单的数据增广：随机翻转、随机亮度调整"""
    X_aug = X.copy()
    N, C, H, W = X_aug.shape
    for i in range(N):
        img = X_aug[i]
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1]
        if np.random.rand() < 0.2:
            img = img[:, ::-1, :]
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 1)
        X_aug[i] = img
    return X_aug

# --------------------------
# 2. Optimizers
# --------------------------
class SGD:
    def __init__(self, lr=1e-2):
        self.lr = lr
    def update(self, W, dW, b=None, db=None):
        W = W - self.lr * dW
        if b is not None:
            b = b - self.lr * db
        return W, b

class Momentum:
    def __init__(self, lr=1e-2, beta=0.9):
        self.lr = lr; self.beta = beta
        self.vW = {}
        self.vb = {}
    def update(self, W, dW, b=None, db=None, key=None):
        # key used to track per-param velocity (string)
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
    def __init__(self, lr=1e-3, rho=0.9, eps=1e-8):
        self.lr = lr; self.rho = rho; self.eps = eps
        self.sW = {}
        self.sb = {}
    def update(self, W, dW, b=None, db=None, key=None):
        if key is None: key = id(W)
        if key not in self.sW:
            self.sW[key] = np.zeros_like(W)
            if b is not None:
                self.sb[key] = np.zeros_like(b)
        self.sW[key] = self.rho*self.sW[key] + (1-self.rho)*(dW**2)
        W = W - self.lr * dW / (np.sqrt(self.sW[key]) + self.eps)
        if b is not None:
            self.sb[key] = self.rho*self.sb[key] + (1-self.rho)*(db**2)
            b = b - self.lr * db / (np.sqrt(self.sb[key]) + self.eps)
        return W, b

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.mW = {}; self.vW = {}; self.mb = {}; self.vb = {}; self.t = {}
    def update(self, W, dW, b=None, db=None, key=None):
        if key is None: key = id(W)
        if key not in self.mW:
            self.mW[key] = np.zeros_like(W); self.vW[key] = np.zeros_like(W)
            self.t[key] = 0
            if b is not None:
                self.mb[key] = np.zeros_like(b); self.vb[key] = np.zeros_like(b)
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

# --------------------------
# 3. Layers
# --------------------------
class Conv2D:
    def __init__(self, in_ch, out_ch, k, stride=1, pad=0, optimizer=None, key=None):
        self.in_ch = in_ch; self.out_ch = out_ch; self.k = k
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = np.random.randn(out_ch, in_ch, k, k).astype(np.float32) * scale
        self.b = np.zeros((out_ch,1), dtype=np.float32)
        self.optimizer = optimizer
        self.key = key if key is not None else f"conv_{id(self)}"

    def forward(self, x):
        # x: (N,C,H,W)
        self.x = x
        N,C,H,W = x.shape
        OC,_,K,_ = self.W.shape
        OH, OW = H - K + 1, W - K + 1
        out = np.zeros((N, OC, OH, OW), dtype=np.float32)
        for n in range(N):
            for oc in range(OC):
                for i in range(OH):
                    for j in range(OW):
                        out[n,oc,i,j] = np.sum(x[n,:,i:i+K,j:j+K]*self.W[oc]) + self.b[oc]
        return out

    def backward(self, grad_out):
        # grad_out: (N, OC, OH, OW)
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
            # default SGD small step if no optimizer
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
        # x: (N,C,H,W)
        self.x = x
        N,C,H,W = x.shape
        out_h, out_w = H // 2, W // 2
        out = np.zeros((N,C,out_h,out_w), dtype=x.dtype)
        self.mask = np.zeros_like(x, dtype=bool)
        for i in range(out_h):
            for j in range(out_w):
                block = x[:,:,2*i:2*i+2,2*j:2*j+2]
                maxv = np.max(block, axis=(2,3), keepdims=True)
                out[:,:,i,j] = maxv.squeeze()
                self.mask[:,:,2*i:2*i+2,2*j:2*j+2] = (block == maxv)
        return out
    def backward(self, grad):
        N,C,H,W = self.x.shape
        dx = np.zeros_like(self.x)
        for i in range(grad.shape[2]):
            for j in range(grad.shape[3]):
                dx[:,:,2*i:2*i+2,2*j:2*j+2] += self.mask[:,:,2*i:2*i+2,2*j:2*j+2] * grad[:,:,i,j][:,:,None,None]
        return dx

class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad):
        return grad.reshape(self.shape)

class Dense:
    def __init__(self, in_dim, out_dim, optimizer=None, key=None):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0/in_dim)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.optimizer = optimizer
        self.key = key if key is not None else f"dense_{id(self)}"
    def forward(self, x):
        self.x = x  # (N, in_dim)
        return x.dot(self.W) + self.b
    def backward(self, grad):
        # grad: (N, out_dim)
        dW = self.x.T.dot(grad)
        db = np.sum(grad, axis=0)
        dx = grad.dot(self.W.T)
        if self.optimizer is not None:
            self.W, self.b = self.optimizer.update(self.W, dW, self.b, db, key=self.key)
        else:
            self.W -= 1e-3 * dW
            self.b -= 1e-3 * db
        return dx

# --------------------------
# 4. BatchNorm (channel-wise for conv, or vector for fc)
# --------------------------
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, optimizer=None, key=None, for_conv=True):
        """
        num_features: for_conv=True -> channels (C)
                      for_conv=False -> feature dim (D) for Dense
        """
        self.for_conv = for_conv
        self.eps = eps; self.momentum = momentum
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
        # x shape: conv -> (N,C,H,W), fc -> (N,D)
        self.x_shape = x.shape
        self.x = x
        self.training = training
        
        if self.for_conv:
            mean = np.mean(x, axis=(0,2,3), keepdims=True)
            var = np.var(x, axis=(0,2,3), keepdims=True)
            if training:
                self.mean = mean; self.var = var
                self.x_hat = (x - mean) / np.sqrt(var + self.eps)
                out = self.gamma * self.x_hat + self.beta
                self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
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
                self.mean = mean; self.var = var
                self.x_hat = (x - mean) / np.sqrt(var + self.eps)
                out = self.gamma * self.x_hat + self.beta
                self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
                return out
            else:
                self.mean = self.running_mean
                self.var = self.running_var
                self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
                return self.gamma * self.x_hat + self.beta

    def backward(self, grad_out):
        # grad_out shape matches x
        if self.for_conv:
            x = self.x = self.x_hat  # careful: self.x_hat stored earlier
            m = np.prod(self.x_shape) / self.x_shape[1]  # N*H*W
            # dx_hat = grad_out * gamma
            dx_hat = grad_out * self.gamma
            dvar = np.sum(dx_hat * (self.x_hat * -0.5) / (self.var + self.eps), axis=(0,2,3), keepdims=True)  # alternate form
            # More stable vectorized derivation:
            dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps)**(-3/2), axis=(0,2,3), keepdims=True)
            dmean = np.sum(dx_hat * (-1.0 / np.sqrt(self.var + self.eps)), axis=(0,2,3), keepdims=True) + dvar * np.sum(-2.0*(self.x - self.mean), axis=(0,2,3), keepdims=True)/m
            dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0*(self.x - self.mean)/m + dmean / m

            # grads for gamma/beta
            dgamma = np.sum(grad_out * self.x_hat, axis=(0,2,3), keepdims=True)
            dbeta = np.sum(grad_out, axis=(0,2,3), keepdims=True)

            # update gamma/beta
            if self.optimizer is not None:
                # optimizer expects 2D/1D sometimes; flatten gamma/beta to arrays
                g_shape = self.gamma.shape
                # use optimizer.update but need to pass key unique for gamma and beta
                g_key = self.key + "_g"; b_key = self.key + "_b"
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
            dmean = np.sum(dx_hat * (-1.0 / np.sqrt(self.var + self.eps)), axis=0, keepdims=True) + dvar * np.sum(-2.0*(self.x - self.mean), axis=0, keepdims=True)/m
            dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0*(self.x - self.mean)/m + dmean / m

            dgamma = np.sum(grad_out * self.x_hat, axis=0, keepdims=True)
            dbeta = np.sum(grad_out, axis=0, keepdims=True)
            if self.optimizer is not None:
                g_key = self.key + "_g"; b_key = self.key + "_b"
                self.gamma, _ = self.optimizer.update(self.gamma, dgamma, None, None, key=g_key)
                self.beta, _ = self.optimizer.update(self.beta, dbeta, None, None, key=b_key)
            else:
                self.gamma -= 1e-3 * dgamma
                self.beta -= 1e-3 * dbeta
            return dx

# --------------------------
# 5. Dropout
# --------------------------
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

# --------------------------
# 6. Softmax + Loss
# --------------------------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss_and_grad(logits, y):
    # logits: (N, num_classes)
    p = softmax(logits)
    N = logits.shape[0]
    loss = -np.mean(np.log(p[np.arange(N), y] + 1e-9))
    grad = p
    grad[np.arange(N), y] -= 1
    grad /= N
    return loss, grad

# --------------------------
# 7. Simple Model Definition
# --------------------------
class SimpleCNN:
    def __init__(self, optimizer_name="adam", lr=1e-3):
        # pick optimizer instance to share across layers (so state preserved)
        if optimizer_name == "sgd":
            opt = SGD(lr=lr)
        elif optimizer_name == "momentum":
            opt = Momentum(lr=lr)
        elif optimizer_name == "rmsprop":
            opt = RMSProp(lr=lr)
        else:
            opt = Adam(lr=lr)

        # architecture: Conv->BN->ReLU->Pool -> Conv->BN->ReLU->Pool -> Flatten -> Dense->BN->ReLU->Dropout -> Dense
        self.layers = [
            Conv2D(3, 16, 3, optimizer=opt, key="conv1"),
            BatchNorm(16, optimizer=opt, for_conv=True, key="bn1"),
            ReLU(),
            MaxPool2x2(),

            Conv2D(16, 32, 3, optimizer=opt, key="conv2"),
            BatchNorm(32, optimizer=opt, for_conv=True, key="bn2"),
            ReLU(),
            MaxPool2x2(),

            Flatten(),
            Dense(32*6*6, 128, optimizer=opt, key="dense1"),
            BatchNorm(128, optimizer=opt, for_conv=False, key="bn3"),
            ReLU(),
            Dropout(p=0.5),

            Dense(128, 10, optimizer=opt, key="dense2")
        ]

    def forward(self, x, training=True):
        out = x
        for layer in self.layers:
            # some layers' forward accept training flag
            if isinstance(layer, BatchNorm) or isinstance(layer, Dropout):
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out)
        return out

    def backward(self, grad):
        # go in reverse; call backward for each
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

# --------------------------
# 8. Training Loop
# --------------------------
import sys

def accuracy(pred, y):
    return np.mean(pred == y)

def train(model, X_train, y_train, X_test, y_test,
          epochs=50, batch_size=128, verbose=True, use_augmentation=True):
    n_train = X_train.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    history = {"loss": [], "val_acc": []}
    for ep in range(epochs):
        t0 = time.time()
        idx = np.random.permutation(n_train)
        X_train_sh = X_train[idx]
        y_train_sh = y_train[idx]
        
        # 应用数据增广
        if use_augmentation:
            X_train_sh = data_augmentation(X_train_sh)
        
        losses = []
        accs = []
        num_batches = 0
        print(f"\nEpoch {ep+1}/{epochs}", flush=True)
        sys.stdout.flush()
        for i in range(0, n_train, batch_size):
            xb = X_train_sh[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            logits = model.forward(xb, training=True)
            loss, grad = cross_entropy_loss_and_grad(logits.reshape(len(xb), -1), yb)
            losses.append(loss)
            model.backward(grad)
            probs = softmax(logits.reshape(len(xb), -1))
            acc = accuracy(probs, yb)
            accs.append(acc)
            num_batches += 1
            
            # 每10个batch显示一次
            if (num_batches % 10) == 0 or (i + batch_size >= n_train):
                avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
                avg_acc = np.mean(accs[-10:]) if len(accs) >= 10 else np.mean(accs)
                print(f"  Batch {num_batches:4d}/{steps_per_epoch} | Loss: {loss:.4f} | Acc: {acc:.4f} | Avg: Loss={avg_loss:.4f} Acc={avg_acc:.4f}", flush=True)
                sys.stdout.flush()
        
        avg_loss = np.mean(losses)
        history["loss"].append(avg_loss)

        print("  Evaluating on test set...", end=' ', flush=True)
        sys.stdout.flush()
        logits_test = model.forward(X_test[:2000], training=False)
        preds = np.argmax(softmax(logits_test.reshape(logits_test.shape[0], -1)), axis=1)
        val_acc = accuracy(preds, y_test[:2000])
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Done | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s", flush=True)
            sys.stdout.flush()

    return history

# --------------------------
# 9. Run (main)
# --------------------------
def main():
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    
    print("\nCreating model...")
    model = SimpleCNN(optimizer_name="adam", lr=1e-3)
    print("Training model...")
    hist = train(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128, use_augmentation=True)

    print("\nTraining completed!")
    print(f"Final test accuracy: {hist['val_acc'][-1]:.4f}")

    # 绘图
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist["loss"], label="train loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.suptitle("Training curves")
    plt.show()
    plt.suptitle("Training curves")
    plt.show()

if __name__ == "__main__":
    main()
