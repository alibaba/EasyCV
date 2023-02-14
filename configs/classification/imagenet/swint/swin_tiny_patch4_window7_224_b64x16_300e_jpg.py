_base_ = '../common/dataset/imagenet_classification.py'

num_classes = 1000
# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    mixup_cfg=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=num_classes),
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='swin_tiny_patch4_window7_224',
        num_classes=num_classes,
    ),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=False))

image_size2 = 224
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=image_size2),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='MMRandAugment',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x * 255) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='MMRandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[x * 255 for x in img_norm_cfg['mean'][::-1]],
        fill_std=[x * 255 for x in img_norm_cfg['std'][::-1]]),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline))

# optimizer
paramwise_options = {
    'norm': dict(weight_decay=0.),
    'bias': dict(weight_decay=0.),
    'absolute_pos_embed': dict(weight_decay=0.),
    'relative_position_bias_table': dict(weight_decay=0.)
}
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 1024 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=dict(max_norm=5.0), update_interval=2)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=30)

# runtime settings
total_epochs = 300
