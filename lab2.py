import torch 
from model.backbone3D.resnext import resnext101

model = resnext101()
model.load_pretrain()

#model_scripted = torch.jit.script(model) # Export to TorchScript
#model_scripted.save('model_scripted.pt')

#torch.save(model, '/home/manh/Projects/My-YOWO/weights/backbone3D/resnext-101-kinetics.pt')

state_dict = torch.load('/home/manh/Projects/My-YOWO/model_scripted.pt')
print(state_dict)