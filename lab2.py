import torch
from model.YOLO2Stream import yolo_v8_m
import torch.nn as nn

a = torch.Tensor(3, 4, 5)
b = a[:, [0, 1], [2, 3]]
print(b.shape)