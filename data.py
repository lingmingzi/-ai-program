import os
import urllib.request
import tarfile
import pickle
import numpy as np

DATASET_DIR = 'Dataset'
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_ARCHIVE = os.path.join(DATASET_DIR, "cifar-10-python.tar.gz")
CIFAR_DIR = os.path.join(DATASET_DIR, "cifar-10-batches-py")

os.makedirs(DATASET_DIR, exist_ok=True)


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
    for i in range(1, 6):
        X, y = load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
        xs.append(X)
        ys.append(y)
    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    X_test, y_test = load_batch(os.path.join(CIFAR_DIR, "test_batch"))
    return X_train, y_train, X_test, y_test
