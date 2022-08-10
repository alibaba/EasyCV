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
    type='MoBY',
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    queue_len=4096,
    momentum=0.99,
    pretrained=False,
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='dynamic_swin_tiny_p4_w7_224',
        num_classes=0,
    ),
    neck=dict(
        type='MoBYMLP',
        in_channels=768,
        hid_channels=4096,
        out_channels=256,
        num_layers=2),
    head=dict(
        type='MoBYMLP',
        in_channels=256,
        hid_channels=4096,
        out_channels=256,
        num_layers=2))

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
    imgs_per_gpu=64,  # total 64*8
    workers_per_gpu=4,
    train=dict(
        type='DaliTFRecordMultiViewDataset',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='data/imagenet_tfrecord/train-*'),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline],
    ))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    trans_weight_decay_set=['backbone'])  # 0.001 for 512
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=5e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 300

# export config
export = dict(export_neck=False)
checkpoint_sync_export = True
