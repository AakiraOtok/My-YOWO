import torch
from model.YOLO2Stream import yolo2stream

model = yolo2stream(num_classes=80, backbone_2D='yolov8_l', backbone_3D='shufflenetv2', fusion_module='CFAM')
dummy_clip = torch.Tensor(1, 3, 16, 224, 224)
out = model(dummy_clip)
for x in out:
    print(x.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số lượng tham số: {total_params}")