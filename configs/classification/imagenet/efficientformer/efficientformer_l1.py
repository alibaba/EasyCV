_base_ = '../../../base.py'
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='EfficientFormer',
        layers=[3, 2, 6, 4],
        embed_dims=[48, 96, 224, 448],
        downsamples=[True, True, True, True],
        vit_num=1,
        fork_feat=False,
        distillation=True,
    ),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        with_fc=False,
        in_channels=448,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth',
            label_smooth=0,
        ),
        num_classes=1000))

data_train_list = 'data/imagenet_raw/meta/train_labeled.txt'
data_train_root = 'data/imagenet_raw/train/'
data_test_list = 'data/imagenet_raw/meta/val_labeled.txt'
data_test_root = 'data/imagenet_raw/val/'
data_all_list = 'data/imagenet_raw/meta/all_labeled.txt'
data_root = 'data/imagenet_raw/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
val_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]

data = dict(
    imgs_per_gpu=128,  # total 1024
    workers_per_gpu=4,
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
        pipeline=val_pipeline))

eval_config = dict(interval=1, gpu_collect=True)
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
optimizer = dict(type='AdamW', lr=2e-3, weight_decay=0.025)
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='constant',
    warmup_iters=5,
    warmup_ratio=5e-4,
    warmup_by_epoch=True,
    by_epoch=True)
checkpoint_config = dict(interval=30)

# runtime settings
total_epochs = 300
