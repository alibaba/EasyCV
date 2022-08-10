_base_ = 'configs/base.py'  # _base_ = 'configs/base.py'

oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',
    buckets=['your_bucket'])

work_dir = 'oss://path/to/work_dir/moby'

# model settings
model = dict(
    type='MoBY',
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    queue_len=4096,
    momentum=0.99,
    pretrained=False,
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     in_channels=3,
    #     out_indices=[4],  # 0: conv-1, x: stage-x
    #     norm_cfg=dict(type='BN')),

    # backbone=dict(
    #     type='ShuffleTransformer',
    #     qkv_bias=True,
    #     drop_path_rate = 0.0,
    #     num_classes = 0),
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',
        # model_name='swin_small_patch4_window7_224',         # bad 16G memory will down
        # model_name='swin_tiny_patch4_window7_224',          # good 768
        # model_name='swin_base_patch4_window7_224_in22k',    # bad 16G memory will down
        # model_name='vit_deit_tiny_distilled_patch16_224',   # good 192,

        # model_name = 'vit_deit_small_distilled_patch16_224', # good 384,
        # model_name='xcit_small_12_p16',   # 384
        # model_name='shuffletrans_tiny_p4_w7_224', #768
        model_name='resnet50',  # 2048
        num_classes=0,
    ),
    neck=dict(
        type='MoBYMLP',
        in_channels=2048,
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
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=4,
    train=dict(
        type='DaliTFRecordMultiViewDataset',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet_tfrecord/train-*',
            cache_path='data/'),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline],
    ))

# optimizer
# optimizer = dict(type='SGD', lr=0.001, weight_decay=0.05, momentum=0.9, set_var_bias_nowd=['backbone'])
optimizer = dict(
    type='AdamW', lr=0.001, weight_decay=0.05,
    set_var_bias_nowd=['backbone'])  # 0.001 for 512
# optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05）#, set_var_bias_nowd=['backbone'])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=5e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=20)

# runtime settings
total_epochs = 100

# export config
# export = dict(export_neck=True)
export = dict(export_neck=False)
checkpoint_sync_export = True
