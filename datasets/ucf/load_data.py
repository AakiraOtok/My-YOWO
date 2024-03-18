import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
from . import transforms
import numpy as np

def UCF_collate_fn(batch_data):
    clips  = []
    boxes  = []
    labels = []
    for b in batch_data:
        clips.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    return clips, boxes, labels


class UCF_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, ann_path, clip_length, sampling_rate, transform=transforms.UCF_transform()):
        self.root_path     = root_path                                        # path to root folder
        self.split_path    = os.path.join(root_path, split_path)              # path to split file
        self.data_path     = os.path.join(root_path, data_path)               # path to data folder
        self.ann_path      = os.path.join(root_path, ann_path)                # path to annotation foler
        self.transform     = transform
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate

        with open(self.split_path, 'r') as f:
            self.lines = f.readlines()

        self.nSample       = len(self.lines)

    def __len__(self):
        return self.nSample
    
    def __getitem__(self, index, get_origin_image=False):
        key_frame_path = self.lines[index].rstrip()                   # e.g : labels/Basketball/v_Basketball_g08_c01/00070.txt
        # for linux, replace '/' by '\' for window 
        split_parts    = key_frame_path.split('/')                    # e.g : ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        key_frame_idx  = int(split_parts[-1].split('.')[-2])          # e.g : 70
        video_name     = split_parts[-2]                              # e.g : v_Basketball_g08_c01
        class_name     = split_parts[1]                               # e.g : Baseketball
        video_path     = os.path.join(self.data_path, class_name, video_name) 
        ann_path       = os.path.join(self.ann_path, class_name, video_name)
        # e.g : /home/manh/Datasets/UCF101-24/ucf24/rgb-images/Basketball/v_Basketball_g08_c01

        path = os.path.join(class_name, video_name) # e.g : Basketball/v_Basketball_g08_c01
        clip        = []
        boxes       = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i*self.sampling_rate

            if cur_frame_idx < 1:
                cur_frame_idx = 1

            # get frame
            cur_frame_path = os.path.join(video_path, '{:05d}.jpg'.format(cur_frame_idx))
            cur_frame      = cv2.imread(cur_frame_path)/255.0
            H, W, C        = cur_frame.shape
            cur_frame      = cv2.resize(cur_frame, (300, 300))
            clip.append(cur_frame)

        if get_origin_image == True:
            key_frame_path     = os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))
            original_image     = cv2.imread(key_frame_path)

        # get annotation for key frame
        ann_file_name = os.path.join(ann_path, '{:05d}.txt'.format(key_frame_idx))
        boxes  = []
        labels = []

        with open(ann_file_name) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip().split(' ')
                label = int(line[0])
                labels.append(label)
                box = [float(line[1])/W, float(line[2])/H, float(line[3])/W, float(line[4])/H] 
                boxes.append(box)

        boxes = np.array(boxes)
        boxes = torch.FloatTensor(boxes)

        labels = np.array(labels)
        labels = torch.FloatTensor(labels)

        # clip   : list of (num_frame) np.array [H, W, C] (BGR order, 0...1)
        # boxes  : tensor of (num_box) tensor [nbox, 4], relative coordinate
        # labels : tensor of (num_box) scalar
        clip, boxes = self.transform(clip, boxes)

        # clip   : tensor [C, numframe, H, W] (RBG order)
        # boxes  : tensor of (num_box) tensor [nbox, 4], relative coordinate
        # labels : tensor of (num_box) scalar
        if get_origin_image == True : 
            return original_image, clip, boxes, labels
        else:
            return clip, boxes, labels