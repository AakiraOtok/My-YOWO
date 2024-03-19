import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt

def warmup_learning_rate(optimizer, epoch, lr):
    lr_init = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init + (lr - lr_init)*epoch/5

def train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(3, 5, 7), acc_grad=16, max_epoch=9):
    torch.backends.cudnn.benchmark = True
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    cur_epoch = 1

    while(cur_epoch <= max_epoch):
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 
            t_batch = time.time()

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            loc, conf = model(batch_clip)

            loss = criterion(loc, conf, dboxes, batch_bboxes, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

            print("epoch : {}, iteration : {}, time = {}, loss = {}".format(cur_epoch, iteration + 1, round(time.time() - t_batch, 2), loss))

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= 0.1

        torch.save(model.state_dict(), r"/home/manh/checkpoint/epoch_" + str(cur_epoch) + ".pth")
        print("Saved model at epoch : {}".format(cur_epoch))
        cur_epoch += 1

from datasets.ucf.load_data import UCF_dataset, UCF_collate_fn
from model.MyYOWO import MyYOWO
from utils.box_utils import MultiBoxLoss

def train_on_UCF(size = 300, version = "original", pretrain_path = None):
    root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    split_path = "trainlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate)
    
    dataloader = data.DataLoader(dataset, 8, True, collate_fn=UCF_collate_fn
                                 , num_workers=6, pin_memory=True)
    
    model = MyYOWO(n_classes = 25)
    
    criterion = MultiBoxLoss(num_classes=25)

    return dataloader, model, criterion

if __name__ == "__main__":
    pretrain_path = None
    dataloader, model, criterion = train_on_UCF(version="original", size=300, pretrain_path=pretrain_path)
    biases     = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer  = optim.SGD(params=[{'params' : biases, 'lr' : 2 * 1e-3}, {'params' : not_biases}], lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(3, 5, 7), max_epoch=9)