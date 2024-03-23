#import torch
#import torch.utils.data as data
#import argparse
#import yaml
#import os
#import cv2
#import pickle
#from datasets.ucf import transforms
#from datasets.ucf.load_data import UCF_dataset
    
#if __name__ == "__main__":
    #root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    #split_path = "trainlist.txt"
    #data_path = "rgb-images"
    #ann_path = "labels"
    #clip_length = 16
    #sampling_rate = 1

    #labels = set()

    #dataset = UCF_dataset(root_path, split_path, data_path, ann_path, clip_length, sampling_rate)
    #print(len(dataset))

#from model.MyYOWO import MyYOWO

#model = MyYOWO()

#total_params = sum(p.numel() for p in model.parameters())
#print("Total parameters:", total_params)

import torch 

a = torch.tensor([-1, 1, 1])
b = torch.tensor([-2, 1, 2])
c = (a + b).clamp(0.)
print(c)