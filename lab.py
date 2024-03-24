import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
from datasets.ucf import transforms
from datasets.ucf.load_data import UCF_dataset
from fvcore.nn import FlopCountAnalysis 
from model.MyYOWO import MyYOWO
    
if __name__ == "__main__":
    root_path = r"H:\Datasets\ucf24"
    split_path = "trainlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path, clip_length, sampling_rate)
    origin_image, clip, bboxes, labels = dataset.__getitem__(5, get_origin_image=True)
    model = MyYOWO(n_classes=25, pretrain_path=r"H:\checkpoint\epoch_5.pth")
    
    model = model.to("cuda")
    clip = clip.unsqueeze(0).to("cuda")
    
    flops = FlopCountAnalysis(model, clip)
    print(flops.total())
    
    




<<<<<<< HEAD
=======
    #dataset = UCF_dataset(root_path, split_path, data_path, ann_path, clip_length, sampling_rate)
    #print(len(dataset))

#from model.MyYOWO import MyYOWO

#model = MyYOWO()

#total_params = sum(p.numel() for p in model.parameters())
#print("Total parameters:", total_params)
>>>>>>> 84ea1e708d7311cf827a82a6570d032c44e740e0
