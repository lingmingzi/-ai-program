import os
import sys
from data import DATASET_DIR, CIFAR_DIR, download_and_extract_cifar

print("=" * 50)
print("CIFAR-10 下载诊断")
print("=" * 50)

print(f"\n1. 检查配置:")
print(f"   DATASET_DIR: {DATASET_DIR}")
print(f"   CIFAR_DIR: {CIFAR_DIR}")
print(f"   当前工作目录: {os.getcwd()}")

print(f"\n2. 检查目录状态:")
print(f"   Dataset 目录存在: {os.path.exists(DATASET_DIR)}")
print(f"   cifar-10-batches-py 存在: {os.path.exists(CIFAR_DIR)}")

print(f"\n3. 检查网络连接和下载权限:")
try:
    print(f"   尝试下载数据...")
    download_and_extract_cifar()
    print(f"   ✓ 下载成功！")
except Exception as e:
    print(f"   ✗ 下载失败:")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {str(e)}")
    sys.exit(1)

print(f"\n4. 检查解压后的文件:")
if os.path.exists(CIFAR_DIR):
    files = os.listdir(CIFAR_DIR)
    print(f"   找到 {len(files)} 个文件:")
    for f in sorted(files):
        print(f"      - {f}")
else:
    print(f"   ✗ cifar-10-batches-py 目录不存在")

print("\n" + "=" * 50)
print("诊断完成")
print("=" * 50)
