from layers import Conv2D, ReLU, MaxPool2x2, Flatten, Dense, BatchNorm, Dropout


class SimpleCNN:
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.conv1 = Conv2D(3, 16, 3, optimizer=optimizer, key="conv1")
        self.bn1 = BatchNorm(16, for_conv=True, optimizer=optimizer, key="bn1")
        self.relu1 = ReLU()
        self.pool1 = MaxPool2x2()

        self.conv2 = Conv2D(16, 32, 3, optimizer=optimizer, key="conv2")
        self.bn2 = BatchNorm(32, for_conv=True, optimizer=optimizer, key="bn2")
        self.relu2 = ReLU()
        self.pool2 = MaxPool2x2()

        self.flatten = Flatten()
        self.dense1 = Dense(32 * 6 * 6, 128, optimizer=optimizer, key="dense1")
        self.bn3 = BatchNorm(128, for_conv=False, optimizer=optimizer, key="bn3")
        self.relu3 = ReLU()
        self.dropout = Dropout(p=0.5)
        self.dense2 = Dense(128, 10, optimizer=optimizer, key="dense2")

    def forward(self, x, training=True):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training=training)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.bn2.forward(x, training=training)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.bn3.forward(x, training=training)
        x = self.relu3.forward(x)
        x = self.dropout.forward(x, training=training)
        x = self.dense2.forward(x)
        return x

    def backward(self, grad):
        grad = self.dense2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.bn3.backward(grad)
        grad = self.dense1.backward(grad)
        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.bn2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)

        return grad
