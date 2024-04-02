import torch
from model.YOLO2Stream import yolo_v8_m
import torch.nn as nn

class test1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model

class test2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.conv1(x)

model = test1(test2())
