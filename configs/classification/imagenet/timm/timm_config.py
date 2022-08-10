_base_ = 'configs/base.py'

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    mixup_cfg=dict(
        mixup_alpha=0.2,
        prob=1.0,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000),
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='vit_base_patch16_224',
        num_classes=1000,
    ),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=False))

load_from = None

data_train_list = './imagenet_raw/meta/train_labeled.txt'
data_train_root = './imagenet_raw/train/'
data_test_list = './imagenet_raw/meta/val_labeled.txt'
data_test_root = './imagenet_raw/val/'
data_all_list = './imagenet_raw/meta/all_labeled.txt'
data_root = './imagenet_raw/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='MMAutoAugment'),
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
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=8,
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
    type='AdamW',
    lr=0.003,
    weight_decay=0.3,
    paramwise_options={
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=dict(max_norm=1.0), update_interval=8)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4,
)
checkpoint_config = dict(interval=30)

# runtime settings
total_epochs = 90
