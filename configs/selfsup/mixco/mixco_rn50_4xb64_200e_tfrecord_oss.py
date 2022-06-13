_base_ = '../../base.py'

# open oss config when using oss
# sync local models and logs to oss
# oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])
# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='your oss hosts',
#     buckets=['your oss buckets'])

# model settings
model = dict(
    type='MIXCO',
    pretrained=False,
    train_preprocess=['randomGrayScale', 'gaussianBlur', 'mixUp'],
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2),
    mixco_head=dict(type='ContrastiveHead', temperature=0.05),
)

dataset_type = 'DaliTFRecordMultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
train_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=224, random_area=(0.2, 1.0)),
    dict(
        type='DaliColorTwist',
        prob=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[224, 224],
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
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline],
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.06, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=20)

# runtime settings
total_epochs = 200
