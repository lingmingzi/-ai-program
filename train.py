import sys
import numpy as np
from utils import cross_entropy_loss_and_grad, accuracy, softmax
from data import data_augmentation


def train(model, X_train, y_train, X_test, y_test, batch_size=100, epochs=50, use_augmentation=True):
    N = len(X_train)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X_train_sh = X_train[idx]
        y_train_sh = y_train[idx]
        
        # 应用数据增广
        if use_augmentation:
            X_train_sh = data_augmentation(X_train_sh)
        
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        print(f"\nEpoch {epoch+1}/{epochs}", flush=True)
        sys.stdout.flush()
        for i in range(0, N, batch_size):
            X_batch = X_train_sh[i:i+batch_size]
            y_batch = y_train_sh[i:i+batch_size]
            logits = model.forward(X_batch, training=True)
            loss, grad = cross_entropy_loss_and_grad(logits, y_batch)
            model.backward(grad)
            probs = softmax(logits)
            acc = accuracy(probs, y_batch)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
            
            # 每5个batch显示一次，或最后一个batch
            if (num_batches % 5) == 0 or (i + batch_size >= N):
                avg_loss_so_far = epoch_loss / num_batches
                avg_acc_so_far = epoch_acc / num_batches
                print(f"  Batch {num_batches:4d} | Loss: {loss:.4f} | Acc: {acc:.4f} | Avg Loss: {avg_loss_so_far:.4f} | Avg Acc: {avg_acc_so_far:.4f}", flush=True)
                sys.stdout.flush()
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        
        print("  Evaluating on test set...", end=' ', flush=True)
        sys.stdout.flush()
        test_logits = model.forward(X_test, training=False)
        test_loss, _ = cross_entropy_loss_and_grad(test_logits, y_test)
        test_probs = softmax(test_logits)
        test_acc = accuracy(test_probs, y_test)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Done | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}", flush=True)
        sys.stdout.flush()

    return history
