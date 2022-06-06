_base_ = '../../base.py'

work_dir = 'work_dir/swav_genet'
num_crops = [2, 6]

# model settings
model = dict(
    type='SWAV',
    pretrained=False,
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    backbone=dict(type='PlainNet', plainnet_struct_idx='normal'),
    neck=dict(
        type='NonLinearNeckSwav',
        in_channels=2560,
        hid_channels=2560,
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
size1 = 192
random_area1 = (0.14, 1.0)
train_pipeline1 = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=size1, random_area=random_area1),
    dict(
        type='DaliColorTwist',
        prob=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
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
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
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
            file_pattern='oss://path/to/data/imagenet-train/train-*',
            # root='data/imagenet_tfrecord/',  # pick one of `file_pattern` and `root&list_file`
            # list_file='data/imagenet_tfrecord/train_list.txt'
        ),
        num_views=num_crops,
        pipelines=[train_pipeline1, train_pipeline2],
    ))

# additional hooks
custom_hooks = [
    dict(
        type='SWAVHook',
        gpu_batch_size=data['imgs_per_gpu'],
        dump_path='data/swav/')
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.6,
    weight_decay=0.000001,
    momentum=0.9,
)

optimizer_config = dict(
    ignore_key=['prototypes'], ignore_key_epoch=[1], update_interval=16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.006,
)
# warmup='linear',
# warmup_iters=10,
# warmup_ratio=0.0001,
# warmup_by_epoch=True)

checkpoint_config = dict(interval=10)

total_epochs = 200
