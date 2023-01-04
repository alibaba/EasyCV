_base_ = ['./fcn_r50-d8_512x512_8xb4_60e_voc12aug.py']

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# dataset settings
dataset_type = 'SegDataset'
data_root = 'data/VOCdevkit/VOC2012/'

train_img_root = data_root + 'JPEGImages'
train_label_root = data_root + 'SegmentationClass'
train_list_file = data_root + 'ImageSets/Segmentation/train.txt'

data = dict(
    train=dict(
        type=dataset_type,
        ignore_index=255,
        data_source=dict(
            _delete_=True,
            type='SegSourceRaw',
            img_root=train_img_root,
            label_root=train_label_root,
            split=train_list_file,
            classes=CLASSES),
    ))
