import os
import pickle
import numpy as np

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, epoch, history=None, filename='model_latest.pkl'):
    """保存模型及训练历史"""
    filepath = os.path.join(MODEL_DIR, filename)
    model_data = {
        'layers': model.layers,
        'epoch': epoch,
        'history': history
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"模型已保存到 {filepath}")


def load_model(model_class, filename='model_latest.pkl'):
    """加载已保存的模型"""
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        print(f"模型文件不存在: {filepath}")
        return None, 0, None
    
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建新模型实例并恢复层
        model = model_class()
        model.layers = model_data['layers']
        epoch = model_data.get('epoch', 0)
        history = model_data.get('history', None)
        print(f"模型已从 {filepath} 加载 (Epoch: {epoch})")
        return model, epoch, history
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, 0, None


def model_exists(filename='model_latest.pkl'):
    """检查模型文件是否存在"""
    filepath = os.path.join(MODEL_DIR, filename)
    return os.path.exists(filepath)
