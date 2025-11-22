import torch

# 确保CUDA可用
assert torch.cuda.is_available(), "CUDA不可用"

# 测试简单矩阵乘法（模拟报错场景）
device = torch.device("cuda:0")
a = torch.randn(128, 64, device=device, dtype=torch.float32)
b = torch.randn(64, 256, device=device, dtype=torch.float32)

try:
    c = torch.mm(a, b)  # 执行矩阵乘法
    print("cuBLAS矩阵乘法测试成功")
except Exception as e:
    print(f"cuBLAS测试失败：{e}")