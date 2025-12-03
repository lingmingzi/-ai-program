import numpy as np
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from data import load_cifar10
from optimizers import Adam
from model import SimpleCNN
from train import train
from model_io import save_model, load_model, model_exists


def main():
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()

    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    
    # 检查是否有已保存的模型
    start_epoch = 0
    history = None
    if model_exists():
        print("\n发现已保存的模型！")
        choice = input("是否加载之前的模型继续训练? (y/n): ").strip().lower()
        if choice == 'y':
            optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999)
            model, start_epoch, history = load_model(lambda: SimpleCNN(optimizer=optimizer))
            if model is not None:
                print(f"从 Epoch {start_epoch} 继续训练...")
            else:
                print("加载失败，重新开始训练...")
                model = SimpleCNN(optimizer=optimizer)
                start_epoch = 0
                history = None
        else:
            optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999)
            model = SimpleCNN(optimizer=optimizer)
    else:
        optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999)
        model = SimpleCNN(optimizer=optimizer)
    
    print("\nCreating model...")
    print("Training model...")
    
    # 如果有之前的历史，继续添加
    if history is None:
        new_history = train(model, X_train, y_train, X_test, y_test, batch_size=100, epochs=50, use_augmentation=True)
    else:
        # 继续训练
        new_history = train(model, X_train, y_train, X_test, y_test, batch_size=100, epochs=50, use_augmentation=True)
        # 合并历史
        for key in new_history:
            if key in history:
                history[key].extend(new_history[key])
            else:
                history[key] = new_history[key]

    # 自动保存模型
    final_history = history if history is not None else new_history
    save_model(model, epoch=50, history=final_history, filename='model_latest.pkl')
    
    print("\nTraining completed!")
    print(f"Final test accuracy: {final_history['test_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
