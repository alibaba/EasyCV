_base_ = 'configs/base.py'

total_epochs = 100

# model settings
model_output_dim = 65536

model = dict(
    type='DINO',
    pretrained=False,
    train_preprocess=[
        'randomGrayScale', 'gaussianBlur', 'solarize'
    ],  # 2+6 view, has different augment pipeline, dino is complex
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',

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
    ),

    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(type='DINONeck', in_dim=384, out_dim=model_output_dim),
    config=dict(
        # dino head setting
        # momentum_teacher = 0.9995, #0.9995 for batchsize=256
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,
        use_tfrecord_input=False,

        # dino loss settding
        out_dim=model_output_dim,
        local_crops_number=8,
        warmup_teacher_temp=0.04,  # temperature for sharp softmax
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        epochs=total_epochs,
    ))

data_train_list = 'imagenet_raw/meta/train.txt'
data_train_root = 'imagenet_raw/'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.4, 1.),
        interpolation=3,  # Image.BICUBIC
    ),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='GaussianBlur', kernel_size=23)],
        p=1.0),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
train_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.4, 1.),
        interpolation=3,  # Image.BICUBIC
    ),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='GaussianBlur', kernel_size=23)],
        p=0.1),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='Solarization', threshold=130)],
        p=0.2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
train_pipeline3 = [
    dict(
        type='RandomResizedCrop',
        size=96,
        scale=(0.05, 0.4),
        interpolation=3,  # Image.BICUBIC
    ),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='GaussianBlur', kernel_size=23)],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
data = dict(
    imgs_per_gpu=32,  # total 32*8=256
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type='MultiViewDataset',
        data_source=dict(
            type='SSLSourceImageList',
            list_file=data_train_list,
            root=data_train_root),
        num_views=[1, 1, 8],
        pipelines=[train_pipeline1, train_pipeline2, train_pipeline3]))

custom_hooks = [
    dict(
        type='DINOHook',
        momentum_teacher=0.996,  # 0.9995 for bs=256
        weight_decay=0.04,
        weight_decay_end=0.4,
    )
]

# optimizer
optimizer = dict(type='AdamW', lr=9.375e-4)

optimizer_config = dict(
    grad_clip=dict(max_norm=3),
    ignore_key=['last_layer'],
    ignore_key_epoch=[1],  # in dino, freeze_last_layer in first 1 epoch
    # update_interval=2
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=10)

load_from = None

# export config  drop dino neck
export = dict(export_neck=False)
# checkpoint_sync_export=True
