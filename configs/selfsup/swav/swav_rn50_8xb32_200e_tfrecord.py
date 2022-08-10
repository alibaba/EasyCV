_base_ = '../../base.py'

# model settings
num_crops = [2, 6]

model = dict(
    type='SWAV',
    pretrained=False,
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(
        type='NonLinearNeckSwav',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=False),
    config=dict(
        # multi crop setting
        num_crops=num_crops,

        # swav setting
        crops_for_assign=[0, 1],
        epsilon=0.05,
        nmb_prototypes=3000,
        sinkhorn_iterations=3,
        temperature=0.1,

        # queue setting
        queue_length=3840,
        epoch_queue_starts=15))

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
size1 = 224
random_area1 = (0.14, 1.0)
train_pipeline1 = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=size1, random_area=random_area1),
    dict(
        type='DaliColorTwist',
        prob=0.8,
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size1, size1],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
size2 = 96
random_area2 = (0.05, 0.14)
train_pipeline2 = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=size2, random_area=random_area2),
    dict(
        type='DaliColorTwist',
        prob=0.8,
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size2, size2],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=2,
    train=dict(
        type='DaliTFRecordMultiViewDataset',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet-tfrecord/train-*',
            # root='data/imagenet_tfrecord/',  # pick one of `file_pattern` and `root&list_file`
            # list_file='data/imagenet_tfrecord/train_list.txt'
        ),
        num_views=num_crops,
        pipelines=[train_pipeline1, train_pipeline2],
    ))

custom_hooks = [
    dict(
        type='SWAVHook',
        gpu_batch_size=data['imgs_per_gpu'],
        dump_path='data/swav/')
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=1.2,
    weight_decay=0.000001,
    momentum=0.9,
    paramwise_options={
        '(bn|gn)(\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })

optimizer_config = dict(
    ignore_key=['prototypes'],
    ignore_key_epoch=[1],
    # update_interval=16
)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0006, by_epoch=False)
# warmup='linear',
# warmup_iters=10,
# warmup_ratio=0.0001,
# warmup_by_epoch=True)

checkpoint_config = dict(interval=10)

total_epochs = 200
load_from = None
resume_from = None

# export config
export = dict(export_neck=True)
checkpoint_sync_export = True
