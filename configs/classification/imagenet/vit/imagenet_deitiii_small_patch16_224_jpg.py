_base_ = './deitiii_base_patch16_192.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='VisionTransformer',
        img_size=[224],
        num_classes=1000,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.05,
        use_layer_scale=True))

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
        type='RandomResizedCrop', size=224, scale=(0.08, 1.0),
        interpolation=3),  # interpolation='bicubic'
    dict(type='RandomHorizontalFlip'),
    dict(type='MMAutoAugment', policies=three_augment_policies),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),
    dict(type='CenterCrop', size=224),
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

eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# optimizer
optimizer = dict(lr=0.004)

lr_config = dict(
    min_lr_ratio=0.00001 / 0.004,
    warmup_ratio=0.000001 / 0.004,
)
