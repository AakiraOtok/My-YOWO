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
from utils.gradflow_check import plot_grad_flow
from utils.util import EMA
from eval import call_eval
import logging
from utils.util import load_yaml_file

config = load_yaml_file()

def warmup_learning_rate(optimizer, step):
    max_step = config['max_step_warmup']
    if (step > max_step):
        return
    lr_init = config['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init * step / max_step
 
def train_model(dataloader, 
                model, 
                criterion, 
                optimizer, 
                adjustlr_schedule=config['adjustlr_schedule'], 
                acc_grad=config['acc_grad'], 
                max_epoch=config['max_epoch']):
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)

    while(cur_epoch <= max_epoch):
        cnt_pram_update = 0
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 
            t_batch = time.time()

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            outputs = model(batch_clip)

            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                nclass = labels.shape[1]
                target = torch.Tensor(nbox, 5 + nclass)
                target[:, 0] = i
                target[:, 1:5] = bboxes
                target[:, 5:] = labels
                targets.append(target)

            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets) / acc_grad
            loss_acc += loss.item()
            loss.backward()
            #plot_grad_flow(model.named_parameters()) #model too large, can't see anything!
            #plt.show()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
                if cur_epoch == 1:
                    warmup_learning_rate(optimizer, cnt_pram_update)

                print("epoch : {}, update : {}, time = {}, loss = {}".format(cur_epoch,  cnt_pram_update, round(time.time() - t_batch, 2), loss_acc))
                loss_acc = 0.0
                #if cnt_pram_update % 500 == 0:
                    #torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epch_{}_update_".format(cur_epoch) + str(cnt_pram_update) + ".pth")

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= 0.5
        
        #          model.state_dict()
        save_path_ema = r"/home/manh/Projects/YOLO2Stream/weights/model_checkpoint/ema_epoch_" + str(cur_epoch) + ".pth"
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path = r"/home/manh/Projects/YOLO2Stream/weights/model_checkpoint/epoch_" + str(cur_epoch) + ".pth"
        torch.save(model.state_dict(), save_path)

        print("Saved model at epoch : {}".format(cur_epoch))

        #log_path = '/home/manh/Projects/YOLO2Stream/training.log'
        #map50, mean_ap = call_eval(save_path)
        #logging.basicConfig(filename=log_path, level=logging.INFO)
        #logging.info('mAP 0.5 : {}, mAP : {}'.format(map50, mean_ap))

        cur_epoch += 1


from datasets.ucf.load_data import UCF_dataset, UCF_collate_fn
from model.YOLO2Stream import yolo2stream
from utils.util import ComputeLoss
from model import testing

def train_on_UCF():
    root_path     = config['data_root']
    split_path    = "trainlist.txt"
    data_path     = "rgb-images"
    ann_path      = "labels"
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate)
    
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=UCF_collate_fn
                                 , num_workers=config['num_workers'], pin_memory=True)
    
    model = yolo2stream(num_classes=config['num_classes'],
                        backbone_2D=config['backbone2D'],
                        backbone_3D=config['backbone3D'],
                        fusion_module=config['fusion_module'],
                        interchannels=config['interchannels'],
                        pretrain_path=config['pretrain_yolo2stream'])

    total_params = round(sum(p.numel() for p in model.net2D.parameters()) // 1e6)
    print(f"Net2D #param: {total_params}")

    total_params = round(sum(p.numel() for p in model.net3D.parameters()) // 1e6)
    print(f"Net3D #param: {total_params}")
    
    total_params = round(sum(p.numel() for p in model.decoupled_head.parameters()) // 1e6)
    print(f"Decoupled head #param: {total_params}")

    total_params = round(sum(p.numel() for p in model.fusion.parameters()) // 1e6)
    print(f"Fusion #param: {total_params}")

    total_params = round(sum(p.numel() for p in model.detection_head.parameters()) // 1e6)
    print(f"Detection #param: {total_params}")

    total_params = round(sum(p.numel() for p in model.parameters()) // 1e6)
    print(f"Tổng số lượng tham số: {total_params}")
    #sys.exit()
    model.train()
    model.to("cuda")
    
    criterion = ComputeLoss(model)

    return dataloader, model, criterion

if __name__ == "__main__":
    dataloader, model, criterion = train_on_UCF()

    optimizer  = optim.AdamW(params=model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])

    train_model(dataloader, model, criterion, optimizer)   