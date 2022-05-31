_base_ = './resnet50_b32x8_100e_jpg.py'

# dataset settings
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
size = 224
train_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=size, random_area=(0.08, 1.0)),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size, size],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
val_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliResize', resize_shorter=size * 1.15),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size, size],
        mean=mean,
        std=std,
        prob=0.0)
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=4,
    train=dict(
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet-tfrecord/train-*',
            # root=data_root,  # pick one of `file_pattern` and `root&list_file`
            # list_file=data_root + 'meta/train.txt',
            cache_path='data/cache/train',
        ),
        pipeline=train_pipeline,
        label_offset=1),
    val=dict(
        imgs_per_gpu=50,
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet-tfrecord/train-*',
            # root=data_root,
            # list_file=data_root + 'meta/train_relative.txt',
            cache_path='data/cache/val',
        ),
        pipeline=val_pipeline,
        random_shuffle=False,
        label_offset=1))
