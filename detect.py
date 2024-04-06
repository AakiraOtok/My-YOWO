
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
from utils.box_utils import draw_bounding_box
from utils.util import non_max_suppression
from datasets.ucf.transforms import UCF_transform, Augmentation

UCF101_idx2name = {
    0  : "Baseketball",
    1  : "BaseketballDunk",
    2  : "Biking",
    3  : "CliffDiving",
    4  : "CricketBowling",
    5  : "Diving", 
    6  : "Fencing",
    7  : "FloorGymnastics",
    8  : "GolfSwing",
    9 : "HorseRiding",
    10 : "IceDancing",
    11 : "LongJump",
    12 : "PoleVault",
    13 : "RopeClimbing",
    14 : "SalsaSpin",
    15 : "SkateBoarding",
    16 : "Skiing",
    17 : "Skijet",
    18 : "Soccer Juggling",
    19 : "Surfing",
    20 : "TennisSwing",
    21 : "TrampolineJumping",
    22 : "VolleyballSpiking",
    23 : "WalkingWithDog"
}


def detect(dataset, model, num_classes=21, mapping=UCF101_idx2name):
    model.to("cuda")
    #for images, bboxes, labels, difficulties in dataloader:
    for idx in range(dataset.__len__()):
        origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)

        clip = clip.unsqueeze(0).to("cuda")
        outputs = model(clip)
        outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]
        #print(outputs[0])
        #sys.exit()
        #print(bboxes)

        origin_image = cv2.resize(origin_image, (224, 224))
        draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)
        cv2.imshow('img', origin_image)
        k = cv2.waitKey()

        if (k == ord('q')):
            break
        #for box in bboxes:
            #pt1 = (int(box[0]*224), int(box[1]*224))
            #pt2 = (int(box[2]*224), int(box[3]*224))
            #print(pt1, pt2)
            #cv2.rectangle(origin_image, pt1, pt2, 1, 1, 1)
            #cv2.imshow("img", origin_image)
            #k = cv2.waitKey()

            #if (k == ord('q')):
                #break
        #cv2.imwrite(r"H:\detect_images\_" + str(idx) + r".jpg", origin_image)
        #print("ok")
        #print("image {} saved!".format(idx))


def detect_on_UCF101(size=300, version="original", pretrain_path=None):
    root_path = '/home/manh/Datasets/UCF101-24/ucf242'
    split_path = "testlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, transform=UCF_transform())

    #model = MyYOWO(n_classes=25, pretrain_path=pretrain_path)
    #model = superYOWO(num_classes=25, pretrain_path=pretrain_path)
    model = yolo_v8(num_classes=24, ver='l', backbone_3D='shufflenetv2', fusion_module='CFAM', pretrain_path=pretrain_path)    
        
    num_classes = 24
    mapping = UCF101_idx2name
    return dataset, model, num_classes, mapping


if __name__ == "__main__":

    pretrain_path = '/home/manh/Projects/YOLO2Stream/weights/model_checkpoint/ema_epoch_2.pth' 
    
    dataset, model, num_classes, mapping = detect_on_UCF101(pretrain_path=pretrain_path, version="FPN", size=300)
    model.eval()
    
    detect(dataset, model, num_classes=num_classes, mapping=mapping)