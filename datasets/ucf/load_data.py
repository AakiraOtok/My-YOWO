import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
from . import transforms

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
        self.ann_path      = os.path.join(root_path, ann_path)                # path to annotation file
        self.transform     = transform
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate

        with open(self.split_path, 'r') as f:
            self.lines = f.readlines()

        with open(self.ann_path, 'rb') as f:
            self.ann_dict = pickle.load(f)

        self.nSample       = len(self.lines)

    def __len__(self):
        return self.nSample
    
    def __getitem__(self, index):
        key_frame_path = self.lines[index].rstrip()                   # e.g : labels/Basketball/v_Basketball_g08_c01/00070.txt
        # for linux, replace '/' by '\' for window 
        split_parts    = key_frame_path.split('/')                    # e.g : ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        key_frame_idx  = int(split_parts[-1].split('.')[-2])          # e.g : 70
        video_name     = split_parts[-2]                              # e.g : v_Basketball_g08_c01
        class_name     = split_parts[1]                               # e.g : Baseketball
        video_path     = os.path.join(self.data_path, class_name, video_name) 
        # e.g : /home/manh/Datasets/UCF101-24/ucf24/rgb-images/Basketball/v_Basketball_g08_c01

        path = os.path.join(class_name, video_name) # e.g : Basketball/v_Basketball_g08_c01
        clip        = []
        boxes       = []
        label       = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i*self.sampling_rate

            if cur_frame_idx < 1:
                cur_frame_idx = 1

            # get frame
            cur_frame_path = os.path.join(video_path, '{:05d}.jpg'.format(cur_frame_idx))
            cur_frame      = cv2.imread(cur_frame_path)/255.0
            H, W, C        = cur_frame.shape
            cur_frame      = cv2.resize(cur_frame, (224, 224))
            clip.append(cur_frame)

        # get annotation for key frame
        clip_ann     = self.ann_dict[path]['annotations'][0]  # {label:int, boxes:np.array [nbox, 4 (x_topleft, y_topleft, w, h)], ef:int, sf:int}'
        start_frame  = clip_ann['sf']

        label        = clip_ann['label']

        boxes  = []
        box          = clip_ann['boxes'][key_frame_idx - start_frame]
        boxes.append(torch.tensor([box[0]/W, box[1]/H, box[2]/W, box[3]/H]))

        # clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0...1)
        # boxes : list of (num_box, in ucf101-24 = 1) tensor [nbox, 4], relative coordinate
        # label : a scalar 
        clip, boxes = self.transform(clip, boxes)

        # clip  : tensor [C, numframe, H, W] (RBG order)
        # boxes : list of (num_box) tensor [nbox, 4], relative coordinate
        # label : a scalar
        return clip, boxes, label