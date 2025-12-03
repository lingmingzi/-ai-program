import numpy as np

# 计算网络各层的输出尺寸
H, W = 32, 32
C = 3

print(f"Input: {H}x{W}x{C}")

# Conv2D (kernel=3, no padding)
H = H - 3 + 1
W = W - 3 + 1
C = 16
print(f"After Conv2D(3→16): {H}x{W}x{C}")

# MaxPool2x2
H = H // 2
W = W // 2
print(f"After MaxPool: {H}x{W}x{C}")

# Conv2D (kernel=3)
H = H - 3 + 1
W = W - 3 + 1
C = 32
print(f"After Conv2D(16→32): {H}x{W}x{C}")

# MaxPool2x2
H = H // 2
W = W // 2
print(f"After MaxPool: {H}x{W}x{C}")

# Flatten
flatten_size = H * W * C
print(f"After Flatten: {flatten_size}")

print(f"\n计算验证: {H} * {W} * {C} = {flatten_size}")
