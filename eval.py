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

from datasets.ucf.load_data import UCF_dataset, UCF_collate_fn
from model.YOLO2Stream import yolo_v8
from utils.util import non_max_suppression
import tqdm
from datasets.ucf.transforms import UCF_transform
from utils import util
from model import testing


@torch.no_grad()
def eval(model, dataloader):

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    for batch_clip, batch_bboxes, batch_labels in p_bar:
        batch_clip = batch_clip.to("cuda")

        targets = []
        for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            target = torch.Tensor(bboxes.shape[0], 6)
            target[:, 0] = i
            target[:, 1] = labels
            target[:, 2:] = bboxes
            targets.append(target)

        targets = torch.cat(targets, dim=0).to("cuda")

        height = 224
        width  = 224

        # Inference
        outputs = model(batch_clip)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = non_max_suppression(outputs, 0.005, 0.5)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            #util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                #tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                #tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                #tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                #tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                #util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap

def eval_on_UCF101(pretrain_path, size):

    root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    split_path = "testlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, img_size=(224, 224), transform=UCF_transform())
    
    dataloader = data.DataLoader(dataset, 32, False, collate_fn=UCF_collate_fn
                                 , num_workers=6, pin_memory=True)
    
    model = yolo_v8(num_classes=24, ver='l', backbone_3D='shufflenetv2', fusion_module='CFAM', pretrain_path=pretrain_path)
    model.to("cuda")
    model.eval()

    return model, dataloader

def call_eval(pretrain_path):
    size          = (224, 224)
    model, dataloader = eval_on_UCF101(pretrain_path=pretrain_path, size=size)
    map50, mean_ap = eval(model, dataloader)
    print(map50)
    print()
    print("=================================================================")
    print()
    print(mean_ap)
    return map50, mean_ap
    

if __name__ == "__main__":

    pretrain_path = '/home/manh/Projects/YOLO2Stream/weights/model_checkpoint/epoch_1.pth'
    size          = (224, 224)

    model, dataloader = eval_on_UCF101(pretrain_path=pretrain_path, size=size)
    map50, mean_ap = eval(model, dataloader)
    print(map50)
    print()
    print("=================================================================")
    print()
    print(mean_ap)
    




