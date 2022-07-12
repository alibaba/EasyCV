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
        layers=[48, 96, 224, 448],
        embed_dims=[3, 2, 6, 4],
        downsamples=[True, True, True, True],
        vit_num=1,
        fork_feat=False,
        distillation=False,
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

data_train_list = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/meta/train_labeled.txt'
data_train_root = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/train/'
data_test_list = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/meta/val_labeled.txt'
data_test_root = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/val/'
data_all_list = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/meta/all_labeled.txt'
data_root = '/apsarapangu/disk2/yunji.cjy/imagenet_raw/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=32,  # total 256
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
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 100
