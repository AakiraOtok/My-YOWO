from datasets.ucf import load_data, transforms

root_path = '/home/manh/Datasets/UCF101-24/ucf24/'
split_path = 'trainlist.txt'
data_path  = 'rgb-images'
ann_path   = 'annotation/pyannot.pkl'
tf         = transforms.UCF_transform()
clip_length = 16
sampling_rate = 1

tt = load_data.UCF_dataset(root_path, split_path, data_path, ann_path, clip_length, sampling_rate, tf)
data = tt.__getitem__(0)

clip, boxes, label = data 