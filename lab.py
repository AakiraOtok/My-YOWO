import torch

path = '/home/manh/Projects/My-YOWO/weights/backbone2D/YOLOv8/yolov8m.pt'
state_dict = torch.load(path)
model = state_dict['model']

print(model.keys())
#print(state_dict.values())
