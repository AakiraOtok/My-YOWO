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
    loss_acc = 0.0

    while(cur_epoch <= max_epoch):
        cnt_pram_update = 0
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 
            t_batch = time.time()

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            loc, conf = model(batch_clip)

            loss = criterion(loc, conf, dboxes, batch_bboxes, batch_labels) / acc_grad
            loss_acc += loss.item()
            loss.backward()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()

                print("epoch : {}, update : {}, time = {}, loss = {}".format(cur_epoch,  cnt_pram_update, round(time.time() - t_batch, 2), loss_acc))
                loss_acc = 0.0
                if cnt_pram_update % 500 == 0:
                    torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epch_{}_update_".format(cur_epoch) + str(cnt_pram_update) + ".pth")

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= 0.5

        torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epoch_" + str(cur_epoch) + ".pth")
        print("Saved model at epoch : {}".format(cur_epoch))
        cur_epoch += 1

from datasets.ucf.load_data import UCF_dataset, UCF_collate_fn
from model.superYOWO import superYOWO
from utils.box_utils import MultiBoxLoss, MultiBox_CIoU_Loss

def train_on_UCF(img_size = (224, 224), version = "original", pretrain_path = None):
    root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    split_path = "trainlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, img_size=img_size)
    
    dataloader = data.DataLoader(dataset, 8, True, collate_fn=UCF_collate_fn
                                 , num_workers=6, pin_memory=True)
    
    model = superYOWO(num_classes = 25, pretrain_path=pretrain_path)
    
    criterion = MultiBox_CIoU_Loss(num_classes=25)
    #criterion = MultiBoxLoss(num_classes=25)

    return dataloader, model, criterion

if __name__ == "__main__":
    pretrain_path = None
    dataloader, model, criterion = train_on_UCF(version="original", img_size=(224, 224), pretrain_path=pretrain_path)
    biases     = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer  = optim.AdamW(params=[{'params' : biases, 'lr' : 2 * 1e-4}, {'params' : not_biases}], lr= 1e-4, weight_decay=5e-4)

    train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(1, 2, 3, 4), max_epoch=7)