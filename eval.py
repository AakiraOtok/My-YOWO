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
from utils.box_utils import Non_Maximum_Suppression, jaccard
from tqdm import tqdm
from model.YOLO2Stream import superYOWO

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    rec = rec.detach().cpu().numpy()
    prec = prec.detach().cpu().numpy()
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p
        ap /= 11
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calc_APs(model, dataset, threshold=0.5, num_classes=21):
    """
    tính APs và mAP của model, trả về 1 tensor gồm [nclass] phần tử
    """

    model.to("cuda")

    # Tạo test dataloader, xử lý từng bức ảnh (batch_size=1)
    #dataloader = VOC_create_test_dataloader(batch_size=1, shuffle=0, data_folder_path=r"H:\projectWPD\data", version=r"VOC2007", id_txt_file=r"test.txt")
    dboxes = model.create_prior_boxes().to("cuda")

    # List chứa [nclass] list thành phần khác, mỗi list thành phần thứ i là các tensor TP/FP/confidence cho class thứ i
    TP     = [[] for _ in range(num_classes)]
    FP     = [[] for _ in range(num_classes)]
    confs  = [[] for _ in range(num_classes)]

    # Tensor [nclass] : tổng số ground truth box của mỗi class
    total  = torch.zeros(num_classes).to("cuda")

    with torch.set_grad_enabled(False):
        #for imgs, targets in tqdm(dataloader):
        for idx_dataset in tqdm(range(len(dataset))):
            img, bboxes, labels = dataset.__getitem__(idx_dataset)
            difficult = torch.Tensor(labels.shape[0])
            difficult.fill_(0)

            # Chuẩn bị data
            img    = img.unsqueeze(0).to("cuda")
            bboxes = bboxes.to("cuda")
            labels = labels.long().to("cuda")
            difficult = difficult.long().to("cuda")

            # Đưa vào mạng
            offset, conf = model(img)

            offset.squeeze_(0) # [1, 8732, 4]  ->  [8732, 4]
            conf.squeeze_(0)   # [1, 8732, num_classes] ->  [8732, num_classes]
            
            # [nbox, 4], [nbox]     ,  [nbox], nếu không có box được tìm thấy thì trả về None
            pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset, conf, conf_threshold=0.005, iou_threshold=0.5, num_classes=num_classes)

            # sort lại theo confidence score
            if (pred_bboxes != None):
                pred_confs, idx = torch.sort(pred_confs, descending=True)
                pred_bboxes     = pred_bboxes[idx]
                pred_labels     = pred_labels[idx]

            for cur_class in range(1, num_classes):
                # lọc các bboxes có có nhãn là class đang xét
                mask = (labels == cur_class)
                cur_bboxes = bboxes[mask]
                cur_difficult = difficult[mask]

                mask = (cur_difficult == 0)
                total[cur_class] += mask.sum()

                # lọc các predict boxes có có nhãn là class đang xét]
                if (pred_bboxes == None):
                    continue
                mask            = (pred_labels == cur_class)
                if (not mask.any()):
                    continue

                cur_pred_bboxes = pred_bboxes[mask]
                cur_pred_confs  = pred_confs[mask]

                cur_TP = torch.zeros(cur_pred_bboxes.size(0)).to("cuda")
                cur_FP = torch.zeros(cur_pred_bboxes.size(0)).to("cuda")
                matched = torch.LongTensor(cur_bboxes.size(0)).fill_(0).to("cuda")

                # Với mỗi pbox, tìm bbox overlap nhiều nhất, nếu max overlap < threshold hoặc bbox đó đã được match thì box đó là FP
                # ngược lại thì match bbox đó với pbox và pbox này là TP
                for pbox in range(cur_pred_bboxes.size(0)):
                    max_iou   = 0
                    best_bbox = 0

                    for bbox in range(cur_bboxes.size(0)):
                        overlap = jaccard(cur_pred_bboxes[pbox:pbox+1], cur_bboxes[bbox:bbox+1])[0, 0].item()
                        if overlap > max_iou:
                            max_iou   = overlap
                            best_bbox = bbox

                    if (max_iou > threshold):
                        if cur_difficult[best_bbox] == 0:
                            if matched[best_bbox] == 0:
                                cur_TP[pbox] = 1
                                matched[best_bbox] = 1
                            else:
                                cur_FP[pbox] = 1
                    else:
                        cur_FP[pbox] = 1

                TP[cur_class].append(cur_TP)
                FP[cur_class].append(cur_FP)
                confs[cur_class].append(cur_pred_confs)

        APs = torch.zeros(num_classes).to("cuda")

        # Tránh chia cho 0
        epsilon = 1e-10
    
        # Tính AP cho mỗi class
        for cur_class in range(1, num_classes):
            if len(TP[cur_class]) == 0:
                continue
            TP[cur_class] = torch.cat(TP[cur_class])
            FP[cur_class] = torch.cat(FP[cur_class])
            confs[cur_class] = torch.cat(confs[cur_class])

            confs[cur_class], idx = torch.sort(confs[cur_class], descending=True)
            TP[cur_class]         = TP[cur_class][idx]
            FP[cur_class]         = FP[cur_class][idx]

            TP_acc = torch.cumsum(TP[cur_class], dim=0)
            FP_acc = torch.cumsum(FP[cur_class], dim=0)

            precision = TP_acc/(TP_acc + FP_acc + epsilon)
            recall    = TP_acc/(total[cur_class] + epsilon)
            precision = torch.cat((torch.tensor([1]).to("cuda"), precision))
            recall    = torch.cat((torch.tensor([0]).to("cuda"), recall))

            #APs[cur_class] = torch.trapz(precision, recall)
            APs[cur_class] = voc_ap(recall, precision)
            
            plt.plot(recall.detach().cpu().numpy(), precision.detach().cpu().numpy())
            plt.show()

        return APs

def eval_on_UCF101(pretrain_path, size=(224, 224)):
    root_path = "/home/manh/Datasets/UCF101-24/ucf242"
    split_path = "testlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate)

    #model = MyYOWO(n_classes=25, pretrain_path=pretrain_path)
    model = superYOWO(num_classes=25, pretrain_path=pretrain_path)
        
    return dataset, model

if __name__ == "__main__":

    pretrain_path = "/home/manh/Projects/My-YOWO/weights/model_checkpoint/epoch_7.pth"
    size          = (224, 224)
    num_classes   = 25

    dataset, model = eval_on_UCF101(pretrain_path=pretrain_path, size=size)

    model.eval()

    APs = calc_APs(model, dataset, num_classes=num_classes)
    #APs = APs[1:] # bỏ background
    print(APs)
    print(APs.mean())


