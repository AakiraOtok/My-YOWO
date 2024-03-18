
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
from model.MyYOWO import MyYOWO
from utils.box_utils import MultiBoxLoss, Non_Maximum_Suppression, draw_bounding_box

UCF101_idx2name = {
    1  : "Baseketball",
    2  : "BaseketballDunk",
    3  : "Biking",
    4  : "CliffDiving",
    5  : "CricketBowling",
    6  : "Diving", 
    7  : "Fencing",
    8  : "FloorGymnastics",
    9  : "GolfSwing",
    10 : "HorseRiding",
    11 : "IceDancing",
    12 : "LongJump",
    13 : "PoleVault",
    14 : "RopeClimbing",
    15 : "SalsaSpin",
    16 : "SkateBoarding",
    17 : "Skiing",
    18 : "Skijet",
    19 : "Soccer Juggling",
    20 : "Surfing",
    21 : "TennisSwing",
    22 : "TrampolineJumping",
    23 : "VolleyballSpiking",
    24 : "WalkingWithDog"
}


def detect(dataset, model, num_classes=21, mapping=UCF101_idx2name):
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    #for images, bboxes, labels, difficulties in dataloader:
    for idx in range(dataset.__len__()):
        origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)

        clip = clip.unsqueeze(0).to("cuda")
        offset, conf = model(clip)
        offset = offset.to("cuda")
        conf   = conf.to("cuda")
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.3, iou_threshold=0.45, top_k=200, num_classes=num_classes)

        draw_bounding_box(origin_image, pred_bboxes, pred_labels, pred_confs, mapping)
        cv2.imshow("img", origin_image)
        k = cv2.waitKey()
        if (k == ord('q')):
            break
        #cv2.imwrite(r"H:\test_img\_" + str(idx) + r".jpg", origin_image)
        print("ok")

def detect_on_UCF101(size=300, version="original", pretrain_path=None):
    root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    split_path = "testlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate)

    model = MyYOWO(n_classes=25, pretrain_path=pretrain_path)
        
    num_classes = 25
    mapping = UCF101_idx2name
    return dataset, model, num_classes, mapping


if __name__ == "__main__":
    pretrain_path = "/home/manh/checkpoint/iteration_23000.pth"
    
    dataset, model, num_classes, mapping = detect_on_UCF101(pretrain_path=pretrain_path, version="FPN", size=300)
    model.eval()
    
    detect(dataset, model, num_classes=num_classes, mapping=mapping)