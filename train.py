import numpy as np
from utils import cross_entropy_loss_and_grad, accuracy, softmax


def train(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=5):
    N = len(X_train)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X_train_sh = X_train[idx]
        y_train_sh = y_train[idx]
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
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
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        test_logits = model.forward(X_test, training=False)
        test_loss, _ = cross_entropy_loss_and_grad(test_logits, y_test)
        test_probs = softmax(test_logits)
        test_acc = accuracy(test_probs, y_test)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    return history
