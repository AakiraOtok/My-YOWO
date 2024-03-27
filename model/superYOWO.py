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
from model.backbone3D.resnext import resnext101
from model.backbone2D.YOLOv8 import yolo_v8_m

from math import sqrt

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, ft):
        return self.relu(self.bn(self.conv(ft)))


class SeparatorBlock(nn.Module):

    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.block1  = ConvBlock(inchannels, inchannels, 3, padding=1, groups=inchannels, bias=False)
        self.block2  = ConvBlock(inchannels, outchannels, 1, bias=False)

    def forward(self, ft):
        return self.block2(self.block1(ft))

class Fusion_Module(nn.Module):

    def __init__(self, inchannels_3D, inchannels_2D):
        super().__init__()
        self.layers1 = SeparatorBlock(inchannels=inchannels_2D+inchannels_3D,outchannels=1024)
        self.layers2 = SeparatorBlock(inchannels=1024,outchannels=512)

    def forward(self, ft_3D, ft_2D):
        _, C_3D, H_3D, W_3D = ft_3D.shape
        _, C_2D, H_2D, W_2D = ft_2D.shape

        assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

        upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
        ft_3D = upsampling(ft_3D)

        ft = torch.cat((ft_3D, ft_2D), dim = 1)
        out = self.layers1(ft)
        out = self.layers2(out)
        return out
    
    def initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

class Detect_Head(nn.Module):

    def __init__(self, num_classes, inchannels=512, num_Box=6):
        super().__init__()
        self.box = nn.Sequential(
            ConvBlock(inchannels, 1024, 3, padding=1),
            ConvBlock(1024, 512, 3, padding=1),
            nn.Conv2d(512, num_Box * 4, kernel_size=1)
        ) 

        self.cls = nn.Sequential(
            ConvBlock(inchannels, 1024, 3, padding=1),
            ConvBlock(1024, 512, 3, padding=1),
            nn.Conv2d(512, num_Box * num_classes, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, ft):
        batch_size = ft.shape[0]

        box = self.box(ft)
        box = box.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        cls = self.cls(ft)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        return box, cls
    
    def initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

class superYOWO(nn.Module):

    def __init__(self, pretrain_path = None, num_classes = 21):
        super().__init__()

        self.num_classes    = num_classes

        self.base_net2D   = yolo_v8_m()
        self.base_net3D   = resnext101()

        self.fusion_module1 = Fusion_Module(inchannels_3D=2048, inchannels_2D=192)
        self.fusion_module2 = Fusion_Module(inchannels_3D=2048, inchannels_2D=384)
        self.fusion_module3 = Fusion_Module(inchannels_3D=2048, inchannels_2D=576)

        self.detect_head1   = Detect_Head(self.num_classes)
        self.detect_head2   = Detect_Head(self.num_classes)
        self.detect_head3   = Detect_Head(self.num_classes)

        if pretrain_path is not None:
            self.load_state_dict(torch.load(pretrain_path))
        else:
            self.base_net2D.load_pretrain()
            self.base_net3D.load_pretrain()
            
            self.fusion_module1.initialize_weights()
            self.fusion_module2.initialize_weights()
            self.fusion_module3.initialize_weights()

            self.detect_head1.initialize_weights()
            self.detect_head2.initialize_weights()
            self.detect_head3.initialize_weights()

    def create_prior_boxes(self):
        """ 
        mỗi box có dạng [cx, cy, w, h] được scale
        """
        # kích thước feature map tương ứng
        fmap_sizes    = [28, 14, 7]
        
        # scale như trong paper và được tính sẵn thay vì công thức
        # lưu ý ở conv4_3, tác giả xét như một trường hợp đặc biệt (scale 0.1):
        # Ở mục 3.1, trang 7 : 
        # "We set default box with scale 0.1 on conv4 3 .... "
        # "For SSD512 model, we add extra conv12 2 for prediction, set smin to 0.15, and 0.07 on conv4 3...""


        box_scales    = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9] 

        
            
        aspect_ratios = [
                [1., 2., 3., 0.5, 0.333],
                [1., 2., 3., 0.5, 0.333],
                [1., 2., 3., 0.5, 0.333],
            ]
        dboxes = []
        
        
        for idx, fmap_size in enumerate(fmap_sizes):
            for i in range(fmap_size):
                for j in range(fmap_size):

                    # lưu ý, cx trong ảnh là trục hoành, do đó j + 0.5 chứ không phải i + 0.5
                    cx = (j + 0.5) / fmap_size
                    cy = (i + 0.5) / fmap_size

                    for aspect_ratio in aspect_ratios[idx]:
                        scale = box_scales[idx]
                        dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

                        if aspect_ratio == 1.:
                            try:
                                scale = sqrt(scale*box_scales[idx + 1])
                            except IndexError:
                                scale = 1.
                            dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

        dboxes = torch.FloatTensor(dboxes)
        
        #dboxes = pascalVOC_style(dboxes)
        dboxes.clamp_(0, 1)
        #dboxes = yolo_style(dboxes)
                
        return dboxes

    def forward(self, clips):
        key_frames = clips[:, :, -1, :, :]

        ft_2Da, ft_2Db, ft_2Dc = self.base_net2D(key_frames)
        ft_3D                  = self.base_net3D(clips)

        ft_a                   = self.fusion_module1(ft_3D, ft_2Da)
        ft_b                   = self.fusion_module2(ft_3D, ft_2Db)
        ft_c                   = self.fusion_module3(ft_3D, ft_2Dc)

        box1, cls1             = self.detect_head1(ft_a)
        box2, cls2             = self.detect_head2(ft_b)
        box3, cls3             = self.detect_head3(ft_c)

        box = torch.cat((box1, box2, box3), dim = 1)
        cls = torch.cat((cls1, cls2, cls3), dim = 1)

        return box, cls


if __name__ == "__main__":

    model = superYOWO()
    total_params = sum(p.numel() for p in model.parameters())
    print("Tổng số lượng tham số của mô hình: ", total_params) 
    dummy_img = torch.Tensor(1, 3, 16, 224, 224)
    box, cls = model(dummy_img)
    print(box.shape)
    print(cls.shape)