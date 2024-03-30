import torch

a = torch.Tensor(2, 3)
a[:, 0] = 5
print(a)