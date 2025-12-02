# CNN CIFAR-10 分类

从零实现的 CNN 模型，用于 CIFAR-10 图像分类。

## 项目结构

```
├── main.py              # 模块化入口点
├── data.py              # CIFAR-10 数据加载
├── layers.py            # 神经网络层（Conv2D, ReLU, Pool, Dense 等）
├── optimizers.py        # 优化器（SGD, Momentum, RMSProp, Adam）
├── model.py             # SimpleCNN 模型定义
├── train.py             # 训练循环
├── utils.py             # 工具函数（softmax, loss, accuracy）
├── my.py                # 完整的单体实现版本
└── Dataset/             # 数据目录（自动下载）
```

## 功能特性

### 网络架构
- **卷积层 (Conv2D)**: 手工实现的二维卷积操作
- **批归一化 (BatchNorm)**: 支持训练和推理阶段
- **激活函数**: ReLU 激活
- **池化层 (MaxPool2x2)**: 2x2 最大池化
- **全连接层 (Dense)**: 完全连接层
- **Dropout**: 正则化防止过拟合

### 优化器
- SGD（随机梯度下降）
- Momentum（动量优化）
- RMSProp（自适应学习率）
- Adam（自适应矩估计）

### 网络拓扑
```
Input (32x32x3)
  ↓
Conv2D(3→16, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  ↓
Conv2D(16→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  ↓
Flatten (1152)
  ↓
Dense(1152→128) → BatchNorm → ReLU → Dropout(0.5)
  ↓
Dense(128→10)
```

## 使用方法

### 方法 1：运行模块化版本
```bash
python main.py
```

### 方法 2：运行完整版本
```bash
python my.py
```

## 依赖项

- numpy
- matplotlib
- scipy (仅用于某些版本)

## 数据下载

首次运行时会自动下载 CIFAR-10 数据集（~170MB）到 `Dataset/` 目录。

## 实现细节

### 从零实现
- 所有卷积和全连接层的前向传播和反向传播都是手工实现
- 支持批处理和小批量梯度下降
- 包含完整的反向传播算法

### 模块化设计
- `data.py`: 独立的数据加载模块
- `layers.py`: 可复用的网络层组件
- `optimizers.py`: 独立的优化器实现
- `train.py`: 通用的训练函数

## 性能

在 CIFAR-10 测试集上的表现（使用 Adam 优化器）：
- 训练 6 个 epoch
- 批大小：128
- 学习率：1e-3

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 推荐使用的模块化入口点 |
| `my.py` | 原始的完整实现，所有代码在一个文件中 |
| `data.py` | CIFAR-10 自动下载和加载 |
| `layers.py` | 网络层实现（Conv, ReLU, Pool, Dense, BatchNorm, Dropout） |
| `optimizers.py` | 优化算法（SGD, Momentum, RMSProp, Adam） |
| `model.py` | SimpleCNN 网络架构定义 |
| `train.py` | 训练循环实现 |
| `utils.py` | 损失函数、激活函数、评估指标 |

## 许可证

MIT License
