from datasets.ucf.load_data import UCF_dataset
import torch 
import sys
import numpy as np
import cv2

if __name__ == "__main__":
    root_path = '/home/manh/Datasets/UCF101-24/ucf242'
    split_path = "testlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate)
    
    for idx in range(16, dataset.__len__()):
        clip, boxes, labels = dataset.__getitem__(idx)
        clip = clip.permute(1, 2, 3, 0)[:, :, :, (2, 1, 0)].contiguous()
        clip = np.array(clip)

        #print(clip.shape)
        #for frame in clip:
            #cv2.imshow('frame', frame)
            #k = cv2.waitKey()
            #if k == ord('q'):
                #break
        
        frame = clip[15]
        for box in boxes:
            pt1 = (int(box[0] * 224), int(box[1] * 224))
            pt2 = (int(box[2] * 224), int(box[3] * 224))
            cv2.rectangle(frame, pt1, pt2, 1, 1, 1)
            cv2.imshow('img', frame)
            cv2.waitKey()
        
        #sys.exit()