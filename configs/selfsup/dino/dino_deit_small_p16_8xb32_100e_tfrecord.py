_base_ = '../../base.py'

total_epochs = 100
load_from = None

# model settings
num_crops = [2, 10]
model_output_dim = 65536

model = dict(
    type='DINO',
    pretrained=False,
    train_preprocess=[
        'randomGrayScale', 'gaussianBlur', 'solarize'
    ],  # 2+6 view, has different augment pipeline, dino is complex
    backbone=dict(
        type='PytorchImageModelWrapper',

        # swin(dynamic)
        # model_name = 'dynamic_swin_tiny_p4_w7_224',
        # model_name = 'dynamic_swin_small_p4_w7_224',
        # model_name = 'dynamic_swin_base_p4_w7_224',

        # deit(224)
        model_name='dynamic_deit_small_p16',

        # xcit(224)
        # model_name='xcit_small_12_p16',
        # model_name='xcit_medium_24_p16',
        # model_name='xcit_large_24_p8',

        # resnet
        # model_name = 'resnet50',
        # model_name = 'resnet18',
        # model_name = 'resnet34',
        # model_name = 'resnet101',
        # num_classes=0,
    ),

    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(type='DINONeck', in_dim=384, out_dim=model_output_dim),
    config=dict(
        # dino head setting
        # momentum_teacher = 0.9995, #0.9995 for batchsize=256
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,
        use_tfrecord_input=True,

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
random_area1 = (0.4, 1.0)
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
random_area2 = (0.05, 0.4)
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
        type='DINOHook',
        momentum_teacher=0.996,
        weight_decay=0.04,
        weight_decay_end=0.4,
    )
]

# optimizer
optimizer = dict(type='AdamW', lr=1e-3)

optimizer_config = dict(
    grad_clip=dict(max_norm=3),
    ignore_key=['last_layer'],
    ignore_key_epoch=[1],  # in dino, freeze_last_layer in first 1 epoch
    update_interval=2)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=5)

# export config
export = dict(export_neck=False)
# checkpoint_sync_export=True
