from . import ucf_config
import torch
import numpy as np

class UCF_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self):
        pass

    def cvt_to_tensor(self, clip):
        clip = np.array(clip)
        clip = torch.FloatTensor(clip)
        clip = clip[:, :, :, (2, 1, 0)].permute(3, 0, 1, 2).contiguous() # BGR -> RGB
        return clip

    def normalize(self, clip, mean=ucf_config.MEAN, std=ucf_config.STD):
        mean  = torch.FloatTensor(mean).view(-1, 1, 1, 1)
        std   = torch.FloatTensor(std).view(-1, 1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, clip, boxes):
        clip = self.cvt_to_tensor(clip)
        clip = self.normalize(clip)
        return clip, boxes

