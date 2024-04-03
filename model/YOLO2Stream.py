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
from model.backbone2D.YOLOv8 import yolo_v8_m
from utils.box_utils import make_anchors

from math import sqrt
import math

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


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


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


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


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            #print(self.box[i](x[i]).shape) [B, 4 * n_dfl_channel, H, W]
            #print(self.cls[i](x[i]).shape) [B, nclass, H, W]
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
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

    def __init__(self, channels_2D, channels_3D, module_name):
        super().__init__()

        layers = []
        for in_channels in channels_2D:
            if module_name == 'CSP':
                layers.append(CSP(in_ch=in_channels + channels_3D, out_ch=in_channels, n=3))
            elif module_name == 'CFAM':
                layers.append(CFAMBlock(in_channels + channels_3D, in_channels))

        self.csp = nn.ModuleList(layers)

    def forward(self, ft_2D, ft_3D):
        _, C_3D, H_3D, W_3D = ft_3D.shape

        fts = []

        for idx, ft2D in enumerate(ft_2D):
            _, C_2D, H_2D, W_2D = ft2D.shape
            assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

            upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
            ft_3D_t = upsampling(ft_3D)
            ft = torch.cat((ft_3D_t, ft2D), dim = 1)
            fts.append(self.csp[idx](ft))
        
        return list(fts)
    
    def init_weights(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.kaiming_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

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
    def __init__(self, width, depth, num_classes, backbone3D, fusion_module, pretrain_yolo=None, pretrain_path=None):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.net3D = backbone3D

        dummy_img3D = torch.zeros(1, 3, 16, 224, 224)
        dummy_img2D = torch.zeros(1, 3, 224, 224)

        out_2D = self.net(dummy_img2D)
        out_3D = self.net3D(dummy_img3D)

        assert out_3D.shape[2] == 1, "output of 3D branch must have D = 1"

        out_channels_2D = [x.shape[1] for x in out_2D]
        out_channels_3D = out_3D.shape[1]

        self.fusion = Fusion_2D_3D(out_channels_2D, out_channels_3D, fusion_module)
        self.fpn = DarkFPN(width, depth)

        self.detection_head = Head(num_classes, (width[3], width[4], width[5]))
        self.detection_head.stride = torch.tensor([224 / x.shape[-2] for x in out_2D])
        self.stride = self.detection_head.stride
        self.detection_head.initialize_biases()

        if pretrain_path is not None:
            self.load_state_dict(torch.load(pretrain_path))
        else : 
            self.load_pretrain(pretrain_yolo=pretrain_yolo)
            self.net3D.load_pretrain()
            self.fusion.init_weights()

    def forward(self, clips):
        key_frames = clips[:, :, -1, :, :]

        ft_2D = self.net(key_frames)
        ft_3D = self.net3D(clips).squeeze(2)
        
        ft = self.fusion(ft_2D, ft_3D)
        ft = self.fpn(ft)

        #     4 coordinate in absolute form, x_center, y_center, w, h, 
        # [B, 4 + num_classes, 1029]
        return self.detection_head(list(ft))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
    def load_pretrain(self, pretrain_yolo):
        state_dict = self.state_dict()

        pretrain_state_dict = torch.load(pretrain_yolo)
        
        for param_name, value in pretrain_state_dict.items():
            if param_name not in state_dict:
                continue
            state_dict[param_name] = value
            
        self.load_state_dict(state_dict)

def yolo_v8(num_classes, ver, backbone_3D, fusion_module, pretrain_path=None):

    assert ver in ['n', 's', 'm', 'l', 'x'], "only suport for n, s, m, l, or x version"
    assert backbone_3D in ['resnext101', 'mobilenetv2', 'shufflenetv2'], "only suport for resnext101, mobilenetv2 or shufflenetv2"
    assert fusion_module in ['CFAM', 'CSP']

    if ver == 'n':
        pretrain_yolo = '/home/manh/Projects/YOLO2Stream/weights/backbone2D/YOLOv8/v8_n.pth'
        depth = [1, 2, 2]
        width = [3, 16, 32, 64, 128, 256]
    elif ver == 's':
        pretrain_yolo = '/home/manh/Projects/YOLO2Stream/weights/backbone2D/YOLOv8/v8_s.pth'
        depth = [1, 2, 2]
        width = [3, 32, 64, 128, 256, 512]
    elif ver == 'm':
        pretrain_yolo = '/home/manh/Projects/YOLO2Stream/weights/backbone2D/YOLOv8/v8_m2.pth'
        depth = [2, 4, 4]
        width = [3, 48, 96, 192, 384, 576]
    elif ver == 'l':
        pretrain_yolo = '/home/manh/Projects/YOLO2Stream/weights/backbone2D/YOLOv8/v8_l.pth'
        depth = [3, 6, 6]
        width = [3, 64, 128, 256, 512, 512]
    elif ver == 'x':
        pretrain_yolo = '/home/manh/Projects/YOLO2Stream/weights/backbone2D/YOLOv8/v8_x.pth'
        depth = [3, 6, 6]
        width = [3, 80, 160, 320, 640, 640]

    if backbone_3D == 'resnext101':
        backbone3D = resnext.resnext101()
    elif backbone_3D == 'mobilenetv2':
        backbone3D = mobilenetv2.get_model()
    elif backbone_3D == 'shufflenetv2':
        backbone3D = shufflenetv2.get_model()

    return YOLO2Stream(width, depth, num_classes, backbone3D, fusion_module, pretrain_yolo, pretrain_path)


if __name__ == "__main__":

    model = yolo_v8(num_classes=25)
    total_params = sum(p.numel() for p in model.parameters())
    print("Tổng số lượng tham số của mô hình: ", total_params) 
    dummy_img = torch.Tensor(1, 3, 16, 224, 224)
    out = model(dummy_img)