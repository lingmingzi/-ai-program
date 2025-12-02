import numpy as np
from data import load_cifar10
from optimizers import Adam
from model import SimpleCNN
from train import train


def main():
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()

    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999)
    print("\nCreating model...")
    model = SimpleCNN(optimizer=optimizer)
    print("Training model...")
    history = train(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=5)

    print("\nTraining completed!")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
