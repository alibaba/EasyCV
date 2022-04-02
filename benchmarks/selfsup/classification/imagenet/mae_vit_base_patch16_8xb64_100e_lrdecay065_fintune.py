_base_ = 'configs/base.py'

# open oss config when using oss
# sync local models and logs to oss
# oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])
# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='your oss hosts',
#     buckets=['your oss buckets'])

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    mixup_cfg=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000),
    pretrained='weights/mae/mae_pretrain_official.pth',
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='dynamic_vit_base_p16',
        num_classes=1000,
        pretrained=True,
        drop_path_rate=0.1,
        global_pool=True),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=False))

data_train_list = 'data/imagenet/meta/train_labeled.txt'
data_train_root = 'data/imagenet/train/'
data_test_list = 'data/imagenet/meta/val_labeled.txt'
data_test_root = 'data/imagenet/validation/'

dataset_type = 'ClsDataset'
train_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        color_jitter=None,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=True),
]
test_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=False,
    ),
]

data = dict(
    imgs_per_gpu=64,
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
update_interval = 2
optimizer_config = dict(update_interval=update_interval)
eff_batch_size = 64 * 8 * update_interval  # 1024
lr_decay = 0.65
optimizer = dict(
    type='AdamW',
    lr=1e-3 * eff_batch_size / 256,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'patch_embed': dict(lr_mult=lr_decay**13),
        '\\.0\\.': dict(lr_mult=lr_decay**12),
        '\\.1\\.': dict(lr_mult=lr_decay**11),
        '\\.2\\.': dict(lr_mult=lr_decay**10),
        '\\.3\\.': dict(lr_mult=lr_decay**9),
        '\\.4\\.': dict(lr_mult=lr_decay**8),
        '\\.5\\.': dict(lr_mult=lr_decay**7),
        '\\.6\\.': dict(lr_mult=lr_decay**6),
        '\\.7\\.': dict(lr_mult=lr_decay**5),
        '\\.8\\.': dict(lr_mult=lr_decay**4),
        '\\.9\\.': dict(lr_mult=lr_decay**3),
        '\\.10\\.': dict(lr_mult=lr_decay**2),
        '\\.11\\.': dict(lr_mult=lr_decay**1),
        'head': dict(lr_mult=1.0)
    })
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True,
    by_epoch=False)

checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 100
