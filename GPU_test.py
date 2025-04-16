import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.device_count())  # 检查可用的 GPU 数量