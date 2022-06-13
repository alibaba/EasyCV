_base_ = 'configs/base.py'

total_epochs = 100
load_from = None
resume_from = None

# oss io config

oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',
    buckets=['your_bucket'])

work_dir = 'oss://path/to/work_dirs/dino_oss/'

# model settings
num_crops = [2, 6]

model_output_dim = 65536

model = dict(
    type='DINO',
    pretrained=False,
    train_preprocess=[
        'randomGrayScale', 'gaussianBlur', 'solarize'
    ],  # 2+6 view, has different augment pipeline, dino is complex
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     in_channels=3,
    #     out_indices=[4],  # 0: conv-1, x: stage-x
    #     norm_cfg=dict(type='SyncBN')
    # ),
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',
        # model_name='swin_small_patch4_window7_224',         # bad 16G memory will down
        # model_name='swin_tiny_patch4_window7_224',          # good 768
        # model_name='swin_base_patch4_window7_224_in22k',    # bad 16G memory will down
        # model_name='vit_deit_tiny_distilled_patch16_224',   # good 192,
        # model_name = 'vit_deit_small_distilled_patch16_224', # good 384,
        model_name='resnet50',
        num_classes=0,
    ),

    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(type='DINONeck', in_dim=2048, out_dim=model_output_dim),
    config=dict(
        # dino head setting
        momentum_teacher=0.9995,
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,

        # dino loss settding
        out_dim=model_output_dim,
        local_crops_number=num_crops[1],
        warmup_teacher_temp=0.04,  # temperature for sharp softmax
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        epochs=total_epochs,
    ))

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
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1),
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
        saturation=0.2,
        hue=0.1),
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
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=2,
    train=dict(
        type='DaliTFRecordMultiViewDataset',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet_tfrecord/train-*',
            cache_path='data/imagenet_tfrecord_test'),
        num_views=num_crops,
        pipelines=[train_pipeline1, train_pipeline2],
    ))

custom_hooks = [
    dict(
        type='DINOHook',
        momentum_teacher=0.996,
        weight_decay=1e-4,
        weight_decay_end=1e-4,
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.06, weight_decay=1e-4, momentum=0.9)

optimizer_config = dict(
    # grad_clip=dict(max_norm=3),
    ignore_key=['last_layer'],
    ignore_key_epoch=[1],  # in dino, freeze_last_layer in first 1 epoch
    update_interval=2)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.01,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=2)

# export config
export = dict(export_neck=False)
checkpoint_sync_export = True
