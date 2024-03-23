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
import math

# adapted from https://inside-machinelearning.com/en/bounding-boxes-python-function/
# một chút sửa đổi để phù hợp với mục đích sử dụng
def box_label(image, box, label=None, color=(100, 0, 0), txt_color=(255, 255, 255)):
  """
  :param image : np array [H, W, C] (BGR)
  :param label : text, default = None
  """
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label is not None:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def draw_bounding_box(image, bboxes, labels, confs, map_labels):
    """
    Vẽ bouding box, đầu và image có thể là tensor hoặc np array tuỳ tình huống, đã thiết kế để xử lý cả 2 trường hợp
    
    :param image, tensor [1, 3, H, W] (RGB) hoặc numpy array [H, W, 3] (BGR)
    :param bboxes, tensor [nbox, 4]
    :param labels, tensor [nbox]
    :param confs, tensor [nbox]
    :param map_labels, dict, do labels là số nên cần map thành nhãn (string)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous()
        image = image.detach().cpu().numpy()
    
    H, W, C = image.shape 
    H -= 1
    W -= 1

    if bboxes is not None:
        for box, label, conf in zip(bboxes, labels, confs):
            box = box.clone().detach()
            box[0] = max(0, int(box[0]*W))
            box[1] = max(0, int(box[1]*H))
            box[2] = min(W, int(box[2]*W))
            box[3] = min(H, int(box[3]*H))
            text    = str(map_labels[label.item()] + " : " + str(round(conf.item()*100, 2)))
            box_label(image, box, text)
        

def pascalVOC_style(boxes):
    """
    Chuyển từ yolo_style sang pascalVOC_style : [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
     
    :param boxes, tensor [nbox, 4]
    
    return:
    boxes tensor [nbox, 4]
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), dim=1)

def yolo_style(boxes):
    """
    Chuyển từ pascalVOC_style sang yolo_style : [xmin, ymin, xmax, ymax] -> [cx, cy, w, h]
      
    :param boxes, tensor [nbox, 4]

    return:
    boxes tensor [nbox, 4]
    """
    return torch.cat(((boxes[:, :2] + boxes[:, 2:])/2, boxes[:, 2:] - boxes[:, :2]), dim=1)

def coco_style(boxes):
    """
    Chuyển từ pascalVOC_style sang yolo_style : [xmin, ymin, xmax, ymax] -> [x, y, w, h]
    :param boxes, tensor [nbox, 4]
    """
    return torch.cat((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]), dim=1)

# encode (giải thích trên hackmd)
def encode_variance(dboxes, bboxes, variances=[0.1, 0.2]):
    return torch.cat(((bboxes[:, :2] - dboxes[:, :2])/(dboxes[:, 2:]*variances[0]), 
                      torch.log(bboxes[:, 2:]/dboxes[:, 2:] + 1e-10)/variances[1]), dim=1)
                    # torch.log có thể bị inf, add thêm eps = 1e-10

# decode (giải thích trên hackmd)
def decode_variance(dboxes, loc, variances=[0.1, 0.2]):
    return torch.cat((loc[:, :2]*dboxes[:, 2:]*variances[0] + dboxes[:, :2],
                     torch.exp(loc[:, 2:]*variances[1])*dboxes[:, 2:]), dim=1)

def intersect(box_a, box_b):
    """
    Trả về diện tích giao nhau của các box
    các box đang ở dạng [xmin, ymin, xmax, ymax] chuẩn hóa [0...1]
    
    :param box_a, tensor [nbox_a, 4]
    :param box_b, tensor [nbox_b, 4]
    
    return:
    inter tensor, [nbox_a, nbox_b]
    """
    A = box_a.size(0)
    B = box_b.size(0)

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, 0)

    return inter[:, :, 0] *inter[:, :, 1]

def jaccard(box_a, box_b):
    """
     Trả về hệ số jaccard của các box
     các box đang ở dạng [xmin, ymin, xmax, ymax] chuẩn hóa [0...1]
      
    :param box_a, tensor [nbox_a, 4]
    :param box_b, tensor [nbox_b, 4]
    
    return:
    overlap [nbox_a, nbox_b]
    """
    A = box_a.size(0)
    B = box_b.size(0)

    temp = box_a[:, 2:] - box_a[:, :2]
    area_box_a = (temp[:, 0]*temp[:, 1]).unsqueeze(1).expand(A, B)
    temp = box_b[:, 2:] - box_b[:, :2]
    area_box_b = (temp[:, 0]*temp[:, 1]).unsqueeze(0).expand(A, B)

    area = area_box_a + area_box_b
    inter = intersect(box_a, box_b)

    return inter/(area - inter)

def matching_strategy_1(dboxes, bboxes, labels, offset_t, labels_t, idx, threshold=0.5, get_encode=True):
    """_summary_

    Args:
        dboxes : [8732, 4] [cx, cy, w, h]
        bboxes : [nbox, 4] [xmin, ymin, xmax, ymax]
        labels : [nbox]
        các tham số còn lại là truyền vào để gán
    """
    overlaps = jaccard(bboxes, pascalVOC_style(dboxes))

    _, best_dboxes_idx            = overlaps.max(dim=1, keepdim=False)
    matched_overlap, matched_idx  = overlaps.max(dim=0, keepdim=False)

    for i in range(best_dboxes_idx.size(0)):
        matched_idx[best_dboxes_idx[i]] = i

    matched_overlap.index_fill_(dim=0, index=best_dboxes_idx, value=1) # đảm bảo rằng giữ lại các box đã được match sau bước phía dưới
    labels_t[idx]                        = labels[matched_idx]
    labels_t[idx][matched_overlap < threshold] = 0
    offset_t[idx]                        =  encode_variance(dboxes, yolo_style(bboxes[matched_idx]))

def matching_strategy_2(dboxes, bboxes, labels, bboxes_t, labels_t, idx, threshold=0.5, get_encode=True):
    """_summary_

    Args:
        dboxes : [8732, 4] [cx, cy, w, h]
        bboxes : [nbox, 4] [xmin, ymin, xmax, ymax]
        labels : [nbox]
        các tham số còn lại là truyền vào để gán
    """
    overlaps = jaccard(bboxes, pascalVOC_style(dboxes))

    _, best_dboxes_idx            = overlaps.max(dim=1, keepdim=False)
    matched_overlap, matched_idx  = overlaps.max(dim=0, keepdim=False)

    for i in range(best_dboxes_idx.size(0)):
        matched_idx[best_dboxes_idx[i]] = i

    matched_overlap.index_fill_(dim=0, index=best_dboxes_idx, value=1) # đảm bảo rằng giữ lại các box đã được match sau bước phía dưới
    labels_t[idx]                        = labels[matched_idx]
    labels_t[idx][matched_overlap < threshold] = 0
    bboxes_t[idx]                        =  bboxes[matched_idx]

class MultiBoxLoss(nn.Module):
    """
    Args:
        offset_p : [batch, 8732, 4] là offset được model dự đoán ra
        conf_p   : [batch, 8732, nclass] là độ tự tin được model dự đoán ra
        dboxes   : [batch, 8732, 4] [cx, cy, w, h] chuẩn hóa [0..1]
        targets  : [nbox, 5], 4 cái đầu là [xmin, ymin, xmax, ymax] chuẩn hóa [0..1], cái cuối là label 
    
    Out :
        loss = L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, offset_p, conf_p, dboxes, batch_bboxes, batch_labels):

        batch_size = offset_p.size(0)
        nbox       = offset_p.size(1)
        offset_t = torch.Tensor(batch_size, nbox, 4).to("cuda")
        labels_t = torch.LongTensor(batch_size, nbox).to("cuda")

        for idx in range(batch_size):
            bboxes    = batch_bboxes[idx]
            labels    = batch_labels[idx]
            matching_strategy_1(dboxes, bboxes, labels, offset_t, labels_t, idx)

        pos_mask = (labels_t != 0)
        neg_mask = (labels_t == 0)

        #location loss
        loss_l = F.smooth_l1_loss(offset_p[pos_mask], offset_t[pos_mask], reduction="sum")

        #confidence loss
        #hard negative mining
        num_pos = pos_mask.sum(dim=1)
        num_neg = torch.clamp(3*num_pos, max=nbox)

        conf_loss = F.cross_entropy(conf_p.view(-1, self.num_classes), labels_t.view(-1), reduction="none")
        conf_loss = conf_loss.view(batch_size, nbox)

        #print(conf_loss.shape)
        #print(pos_mask.shape)
        #print(conf_loss[pos_mask].sum())
        #sys.exit()
        #print(pos_mask.any())
        
        pos_loss_c          = conf_loss[pos_mask].sum()
        conf_loss           = conf_loss.clone()

        conf_loss[pos_mask] = 0
        _, idx              = torch.sort(conf_loss, dim=1, descending=True)
        _, idx              = torch.sort(idx, dim=1)
        neg_mask            = idx < num_neg.unsqueeze(-1)
        neg_loss_c          = conf_loss[neg_mask].sum()
        loss_c = pos_loss_c + neg_loss_c

        return (loss_l + loss_c)/(num_pos.sum() + 1e-10)
    
def pairwise_jaccard(bboxes_a, bboxes_b):
    b1_xmin = bboxes_a[:, 0]
    b1_ymin = bboxes_a[:, 1]
    b1_xmax = bboxes_a[:, 2]
    b1_ymax = bboxes_a[:, 3]
    b1_w    = b1_xmax - b1_xmin
    b1_h    = b1_ymax - b1_ymin
    b1_area = b1_w * b1_h

    b2_xmin = bboxes_b[:, 0]
    b2_ymin = bboxes_b[:, 1]
    b2_xmax = bboxes_b[:, 2]
    b2_ymax = bboxes_b[:, 3]
    b2_w    = b2_xmax - b2_xmin
    b2_h    = b2_ymax - b2_ymin
    b2_area = b2_w * b2_h

    b3_xmin = torch.max(b1_xmin, b2_xmin)
    b3_xmax = torch.min(b1_xmax, b2_xmax)
    b3_ymin = torch.max(b1_ymin, b2_ymin)
    b3_ymax = torch.min(b1_ymax, b2_ymax)
    b3_w    = (b3_xmax - b3_xmin).clamp(0.)
    b3_h    = (b3_ymax - b3_ymin).clamp(0.)
    b3_area = b3_w * b3_h

    return b3_area / (b1_area + b2_area - b3_area + 1e-7)
    
class MultiBox_CIoU_Loss(nn.Module):
    """
    Args:
        offset_p : [batch, 8732, 4] là offset được model dự đoán ra
        conf_p   : [batch, 8732, nclass] là độ tự tin được model dự đoán ra
        dboxes   : [batch, 8732, 4] [cx, cy, w, h] chuẩn hóa [0..1]
        targets  : [nbox, 5], 4 cái đầu là [xmin, ymin, xmax, ymax] chuẩn hóa [0..1], cái cuối là label 
    
    Out :
        loss = L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_l = 5.
        self.alpha_c = 0.5

    def forward(self, offset_p, conf_p, dboxes, batch_bboxes, batch_labels):

        batch_size = offset_p.size(0)
        nbox       = offset_p.size(1)
        bboxes_t = torch.Tensor(batch_size, nbox, 4).to("cuda")
        bboxes_p = torch.Tensor(batch_size, nbox, 4).to("cuda")
        labels_t = torch.LongTensor(batch_size, nbox).to("cuda")

        for idx in range(batch_size):
            bboxes    = batch_bboxes[idx]
            labels    = batch_labels[idx]
            matching_strategy_2(dboxes, bboxes, labels, bboxes_t, labels_t, idx)
            bboxes_p[idx] = pascalVOC_style(decode_variance(dboxes, offset_p[idx]))

        pos_mask = (labels_t != 0)
        neg_mask = (labels_t == 0)

        #location loss : CIoU
        eps = 1e-7

        bboxes_t = bboxes_t[pos_mask]
        bboxes_p = bboxes_p[pos_mask]

        b1_xmin  = bboxes_t[:, 0]
        b1_ymin  = bboxes_t[:, 1]
        b1_xmax  = bboxes_t[:, 2]
        b1_ymax  = bboxes_t[:, 3]
        b1_w     = b1_xmax - b1_xmin 
        b1_h     = b1_ymax - b1_ymin + eps

        b2_xmin  = bboxes_p[:, 0]
        b2_ymin  = bboxes_p[:, 1]
        b2_xmax  = bboxes_p[:, 2]
        b2_ymax  = bboxes_p[:, 3]
        b2_w     = b2_xmax - b2_xmin 
        b2_h     = b2_ymax - b2_ymin + eps

        IoU = pairwise_jaccard(bboxes_p, bboxes_t)
        if (torch.isnan(IoU).any()):
            print("IoU")
            sys.exit()

        cw  = torch.max(b1_xmax, b2_xmax) - torch.min(b1_xmin, b2_xmin)
        ch  = torch.max(b1_ymax, b2_ymax) - torch.min(b1_ymin, b2_ymin)

        c2   = cw ** 2 + ch ** 2 + eps
        if (torch.isnan(c2).any()):
            print("c2")
            sys.exit()


        rho2 = ((b1_xmax + b1_xmin - b2_xmax - b2_xmin)**2 - (b1_ymax + b1_ymin - b2_ymax - b2_ymin)**2)/4
        if (torch.isnan(rho2).any()):
            print("rho2")
            sys.exit()


        v    = (4 / math.pi ** 2) * (torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h)).pow(2)
        if (torch.isnan(v).any()):
            print("v")
            sys.exit()

        with torch.no_grad():
            alpha = v / (v - IoU + (1 + eps))
        if (torch.isnan(alpha).any()):
            print("alpha")
            sys.exit()

        loss_l = 1.0 - IoU + rho2 / c2 + v * alpha
        loss_l = torch.sum(loss_l) * self.alpha_l

        #confidence loss
        #hard negative mining
        num_pos = pos_mask.sum(dim=1)
        num_neg = torch.clamp(3*num_pos, max=nbox)

        conf_loss = F.cross_entropy(conf_p.view(-1, self.num_classes), labels_t.view(-1), reduction="none")
        conf_loss = conf_loss.view(batch_size, nbox)

        #print(conf_loss.shape)
        #print(pos_mask.shape)
        #print(conf_loss[pos_mask].sum())
        #sys.exit()
        #print(pos_mask.any())
        
        pos_loss_c          = conf_loss[pos_mask].sum()
        conf_loss           = conf_loss.clone()

        conf_loss[pos_mask] = 0
        _, idx              = torch.sort(conf_loss, dim=1, descending=True)
        _, idx              = torch.sort(idx, dim=1)
        neg_mask            = idx < num_neg.unsqueeze(-1)
        neg_loss_c          = conf_loss[neg_mask].sum()
        loss_c = (pos_loss_c + neg_loss_c) * self.alpha_c

        #print(torch.max(v).item(), torch.max(IoU).item())
        print(loss_c.item(), loss_l.item())

        return (loss_l + loss_c)/(num_pos.sum() + 1e-10)

def nms(bboxes, conf, conf_threshold=0.01, iou_threshold=0.45):
    #"""
    #Thực hiện thuật toán non maximum sppression
    #Đầu vào :
     #bboxes : [8732, 4]
     #conf   : [8732]  confidence score
    #Đầu ra :
     #ret_bboxes [nbox, 4]
     #ret_labels [nbox]

     #tất cả đều là tensor
    #"""

    # Giữ lại những box có confidence score > threshold
    mask   = (conf > conf_threshold)
    bboxes = bboxes[mask]
    conf   = conf[mask]

    conf, idx = torch.sort(conf, descending=True)
    bboxes    = bboxes[idx]

    ret_bboxes  = []
    ret_conf    = []

    while(conf.size(0)):
        ret_bboxes.append(bboxes[0])
        ret_conf.append(conf[0:1])

        if conf.size(0) == 1:
            break

        overlap = jaccard(bboxes[0:1], bboxes[1:])
        overlap.squeeze_(0)

        bboxes = bboxes[1:]
        conf   = conf[1:]

        mask   = (overlap < iou_threshold)
        bboxes = bboxes[mask]
        conf   = conf[mask]

    if (len(ret_bboxes) == 0):
        return None, None
    
    return torch.stack(ret_bboxes, dim=0), torch.cat(ret_conf)

def Non_Maximum_Suppression(dboxes, offset, conf, conf_threshold=0.01, iou_threshold=0.45, top_k=200, num_classes=21):
    #"""
    #Thực hiện thuật toán non maximum sppression
    #Đầu vào :
     #loc  : [8732, 4] là offset
     #conf : [8732, config.num_classes]
     #dboxes : [8732, 4] [x, y, w, h]
    #Đầu ra :
     #ret_bboxes [nbox, 4]
     #ret_labels [nbox]
     #ret_confs  [nbox]

     #tất cả đều là tensor
    #"""

    # Lấy bboxes từ dboxes và offset
    bboxes = decode_variance(dboxes, offset)
    bboxes = pascalVOC_style(bboxes)

    # Lấy softmax để tổng xác suất confidence mỗi box = 1
    conf   = F.softmax(conf, dim=1)

    pred_bboxes = [] 
    pred_labels = []
    pred_confs  = []

    for cur_class in range(1, num_classes): # bỏ class 0 là background
        nms_bboxes, nms_conf = nms(bboxes, conf[:, cur_class], conf_threshold, iou_threshold)
        if (nms_bboxes == None):
            continue
        nms_labels           = torch.LongTensor(nms_conf.size(0)).fill_(cur_class).to("cuda")

        pred_bboxes.append(nms_bboxes)
        pred_labels.append(nms_labels)
        pred_confs.append(nms_conf)

    if (len(pred_bboxes) == 0):
        return None, None, None

    pred_bboxes = torch.cat(pred_bboxes)
    pred_labels = torch.cat(pred_labels)
    pred_confs  = torch.cat(pred_confs)

    pred_confs, idx = torch.sort(pred_confs, descending=True)
    pred_bboxes     = pred_bboxes[idx]
    pred_labels     = pred_labels[idx]

    pred_bboxes = pred_bboxes[:top_k]
    pred_labels = pred_labels[:top_k]
    pred_confs  = pred_confs[:top_k]

    return pred_bboxes, pred_labels, pred_confs

