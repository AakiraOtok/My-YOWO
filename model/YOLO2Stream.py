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
from model.backbone3D import resnext, shufflenetv2, mobilenetv2
from model.backbone2D.YOLOv8 import yolo_v8_l, yolo_v8_m, yolo_v8_n, yolo_v8_s, yolo_v8_x
from utils.box_utils import make_anchors

from math import sqrt
import math

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))

class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        # x.shape = [B, 4 * n_dfl_channel, 1029]
        b, c, a = x.shape

        # x.shape = [B, n_dfl_channel, 4, 1029]
        x = x.view(b, 4, self.ch, a).transpose(2, 1)

        return self.conv(x.softmax(1)).view(b, 4, a)
    
class DecoupleHead(torch.nn.Module):

    def __init__(self, interchannels, filters=()):
        super().__init__()
        self.nl = len(filters)  # number of detection layers

        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           Conv(interchannels, interchannels, 3)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           Conv(interchannels, interchannels, 3)) for x in filters)

    def forward(self, x):
        out = []
        for i in range(self.nl):
            #print(self.box[i](x[i]).shape) [B, 4 * n_dfl_channel, H, W]
            #print(self.cls[i](x[i]).shape) [B, nclass, H, W]
            out.append([self.box[i](x[i]), self.cls[i](x[i])])

        return out



class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc, interchannels, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        #c1 = max(filters[0], self.nc)
        #c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        #self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           #Conv(interchannels, interchannels, 3),
                                                           #nn.Conv2d(interchannels, self.nc, 1)) for x in filters)
        #self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           #Conv(interchannels, interchannels, 3),
                                                           #nn.Conv2d(interchannels, 4 * self.ch, 1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(nn.Conv2d(x, self.nc, 3, padding=1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(nn.Conv2d(x, 4 * self.ch, 3, padding=1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            #print(self.box[i](x[i]).shape) [B, 4 * n_dfl_channel, H, W]
            #print(self.cls[i](x[i]).shape) [B, nclass, H, W]
            x[i] = torch.cat((self.box[i](x[i][0]), self.cls[i](x[i][1])), 1)
        if self.training:
            return x
        #                                     transpose !!!      return [1029, 2] and [1029, 1], 1029 = 28*28 + 14*14 + 7*7
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        # [B, nclass + 4 * n_dfl_channel, 1029]
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        
        # [B, 4 * n_dfl_channel, 1029] [B, num_classes, 1029]
        box, cls = x.split((self.ch * 4, self.nc), 1) 

        # print(self.dfl(box).shape) = [B, 4, 1029] -> after decode
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)


        #                 [B, 4, 1029]        [B, num_classes, 1029]
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (224 / s) ** 2)


class Fusion_2D_3D(nn.Module):

    def __init__(self, channels_2D, channels_3D, interchannels, module_name):
        super().__init__()

        box = []
        cls = []
        for in_channels in channels_2D:
            if module_name == 'CSP':
                box.append(CSP(in_ch=in_channels[0] + channels_3D, out_ch=interchannels, n=3))
                cls.append(CSP(in_ch=in_channels[1] + channels_3D, out_ch=interchannels, n=3))
            elif module_name == 'CFAM':
                box.append(CFAMBlock(in_channels[0] + channels_3D, interchannels))
                cls.append(CFAMBlock(in_channels[1] + channels_3D, interchannels))

        self.box = nn.ModuleList(box)
        self.cls = nn.ModuleList(cls)

    def forward(self, ft_2D, ft_3D):
        _, C_3D, H_3D, W_3D = ft_3D.shape

        fts = []

        for idx, ft2D in enumerate(ft_2D):
            _, C_2D, H_2D, W_2D = ft2D[0].shape
            assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

            upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
            ft_3D_t = upsampling(ft_3D)
            ft_box = torch.cat((ft_3D_t, ft2D[0]), dim = 1)
            ft_cls = torch.cat((ft_3D_t, ft2D[1]), dim = 1)
            fts.append([self.box[idx](ft_box), self.cls[idx](ft_cls)])
        
        return fts

class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CFAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        inter_channels = out_channels
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.SiLU())
        
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.SiLU())

        self.sc = CAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.SiLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output

class YOLO2Stream(torch.nn.Module):
    def __init__(self, num_classes, backbone2D, backbone3D, fusion_module, interchannels, pretrain_path=None):
        super().__init__()

        self.inter_channels_decoupled = interchannels[0] 
        self.inter_channels_fusion    = interchannels[1]
        self.inter_channels_detection = interchannels[2]

        self.net2D = backbone2D
        self.net3D = backbone3D

        dummy_img3D = torch.zeros(1, 3, 16, 224, 224)
        dummy_img2D = torch.zeros(1, 3, 224, 224)

        out_2D = self.net2D(dummy_img2D)
        out_3D = self.net3D(dummy_img3D)

        assert out_3D.shape[2] == 1, "output of 3D branch must have D = 1"

        out_channels_2D = [x.shape[1] for x in out_2D]
        out_channels_3D = out_3D.shape[1]

        self.decoupled_head = DecoupleHead(self.inter_channels_decoupled, out_channels_2D)

        # [[box, cls], [box, cls], [box, cls]]
        out_2D = self.decoupled_head(out_2D)

        out_channels_2D = [[x[0].shape[1], x[1].shape[1]] for x in out_2D]                
        self.fusion = Fusion_2D_3D(out_channels_2D, out_channels_3D, self.inter_channels_fusion, fusion_module)

        self.detection_head = Head(num_classes, self.inter_channels_detection, [self.inter_channels_fusion for x in range(len(out_channels_2D))])
        self.detection_head.stride = torch.tensor([224 / x[0].shape[-2] for x in out_2D])
        self.stride = self.detection_head.stride
        self.detection_head.initialize_biases()

        if pretrain_path  != "None":
            self.load_state_dict(torch.load(pretrain_path))
        else : 
            self.net2D.load_pretrain()
            self.net3D.load_pretrain()
            self.init_conv2d()

    def forward(self, clips):
        key_frames = clips[:, :, -1, :, :]

        ft_2D = self.net2D(key_frames)
        ft_3D = self.net3D(clips).squeeze(2)
        
        ft_2D = self.decoupled_head(ft_2D)

        ft = self.fusion(ft_2D, ft_3D)

        # [B, 4 + num_classes, 1029]
        return self.detection_head(list(ft))
    
    def load_pretrain(self, pretrain_yolo):
        state_dict = self.state_dict()

        pretrain_state_dict = torch.load(pretrain_yolo)
        
        for param_name, value in pretrain_state_dict.items():
            if param_name not in state_dict:
                continue
            state_dict[param_name] = value
            
        self.load_state_dict(state_dict)
    
    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        #for c in self.decoupled_head.modules():
            #if isinstance(c, nn.Conv2d):
                #nn.init.kaiming_normal_(c.weight)
                #if c.bias is not None:
                    #nn.init.constant_(c.bias, 0.)

        #for c in self.fusion.modules():
            #if isinstance(c, nn.Conv2d):
                #nn.init.kaiming_normal_(c.weight)
                #if c.bias is not None:
                    #nn.init.constant_(c.bias, 0.)

def yolo2stream(num_classes, backbone_2D, backbone_3D, fusion_module, interchannels, pretrain_path=None):

    assert backbone_2D in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x'], "only suport for n, s, m, l, or x version"
    assert backbone_3D in ['resnext101', 'mobilenetv2', 'shufflenetv2'], "only suport for resnext101, mobilenetv2 or shufflenetv2"
    assert fusion_module in ['CFAM', 'CSP']

    if backbone_2D == 'yolov8_n':
        backbone2D = yolo_v8_n()
    elif backbone_2D == 'yolov8_s':
        backbone2D = yolo_v8_s()
    elif backbone_2D == 'yolov8_m':
        backbone2D = yolo_v8_m()
    elif backbone_2D == 'yolov8_l':
        backbone2D = yolo_v8_l()
    elif backbone_2D == 'yolov8_x':
        backbone2D = yolo_v8_x()

    if backbone_3D == 'resnext101':
        backbone3D = resnext.resnext101()
    elif backbone_3D == 'mobilenetv2':
        backbone3D = mobilenetv2.get_model()
    elif backbone_3D == 'shufflenetv2':
        backbone3D = shufflenetv2.get_model()

    return YOLO2Stream(num_classes, backbone2D, backbone3D, fusion_module, interchannels, pretrain_path)


if __name__ == "__main__":

    model = yolo_v8(num_classes=25)
    total_params = sum(p.numel() for p in model.parameters())
    print("Tổng số lượng tham số của mô hình: ", total_params) 
    dummy_img = torch.Tensor(1, 3, 16, 224, 224)
    out = model(dummy_img)