# from PIL import Image

_base_ = 'configs/base.py'

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    pretrained=False,
    mixup_cfg=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.0,
        num_classes=1000),
    backbone=dict(
        type='VisionTransformer',
        img_size=[192],
        num_classes=1000,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.2,
        use_layer_scale=True),
    head=dict(
        type='ClsHead',
        loss_config=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            label_ceil=True),
        with_fc=False,
        use_num_classes=False))

data_train_list = 'data/imagenet1k/train.txt'
data_train_root = 'data/imagenet1k/train/'
data_test_list = 'data/imagenet1k/val.txt'
data_test_root = 'data/imagenet1k/val/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
three_augment_policies = [[
    dict(type='PILGaussianBlur', prob=1.0, radius_min=0.1, radius_max=2.0),
], [
    dict(type='Solarization', threshold=128),
], [
    dict(type='Grayscale', num_output_channels=3),
]]
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=192, scale=(0.08, 1.0),
        interpolation=3),  # interpolation='bicubic'
    dict(type='RandomHorizontalFlip'),
    dict(type='MMAutoAugment', policies=three_augment_policies),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
size = int((256 / 224) * 192)
test_pipeline = [
    dict(type='Resize', size=size, interpolation=3),
    dict(type='CenterCrop', size=192),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=8,
    use_repeated_augment_sampler=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            type='ClsSourceImageList'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type='ClsSourceImageList'),
        pipeline=test_pipeline))

eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# additional hooks
custom_hooks = []

# optimizer
optimizer = dict(
    type='Lamb',
    lr=0.003,
    weight_decay=0.05,
    eps=1e-8,
    paramwise_options={
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'gamma_1': dict(weight_decay=0.),
        'gamma_2': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None, update_interval=1)

lr_config = dict(
    policy='CosineAnnealingWarmupByEpoch',
    by_epoch=True,
    min_lr_ratio=0.00001 / 0.003,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.000001 / 0.003,
)
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 800

ema = dict(decay=0.99996)
