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
        switch_prob = 0.5,
        mode='batch',
        label_smoothing=0.0,
        num_classes=1000),
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='deitiii_base_p16_192',
        num_classes=1000,
    ),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'BinaryCrossEntropyWithLogitsLoss',
        },
        with_fc=False))

data_train_list = '../../../../dev/imagenet1k/ILSVRC2012_img_val_2/imagenet/train.txt'
data_train_root = '../../../../dev/shm/imagenet1K/ILSVRC2012_img_train/'
data_test_list = '../../../../dev/imagenet1k/ILSVRC2012_img_val_2/imagenet/val.txt'
data_test_root = '../../../../dev/imagenet1k/ILSVRC2012_img_val_2/imagenet/'
# data_all_list = 'imagenet_raw/meta/all_labeled.txt'
# data_root = 'imagenet_raw/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=192, scale=(0.08, 1.0), interpolation=3), # interpolation='bicubic'
    dict(type='RandomHorizontalFlip'),
    dict(type='ThreeAugment'),
    dict(type='ColorJitter',
        brightness=0.3,
        contrast=0.3,
        saturation=0.3),
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
    repeated_aug=True,
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

        'patch_embed.proj.bias': dict(weight_decay=0.),

        'blocks.0.gamma_1': dict(weight_decay=0.),
        'blocks.0.gamma_2': dict(weight_decay=0.),
        'blocks.0.norm1.weight': dict(weight_decay=0.),
        'blocks.0.norm1.bias': dict(weight_decay=0.),
        'blocks.0.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.0.attn.proj.bias': dict(weight_decay=0.),
        'blocks.0.norm2.weight': dict(weight_decay=0.),
        'blocks.0.norm2.bias': dict(weight_decay=0.),
        'blocks.0.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.0.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.1.gamma_1': dict(weight_decay=0.),
        'blocks.1.gamma_2': dict(weight_decay=0.),
        'blocks.1.norm1.weight': dict(weight_decay=0.),
        'blocks.1.norm1.bias': dict(weight_decay=0.),
        'blocks.1.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.1.attn.proj.bias': dict(weight_decay=0.),
        'blocks.1.norm2.weight': dict(weight_decay=0.),
        'blocks.1.norm2.bias': dict(weight_decay=0.),
        'blocks.1.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.1.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.2.gamma_1': dict(weight_decay=0.),
        'blocks.2.gamma_2': dict(weight_decay=0.),
        'blocks.2.norm1.weight': dict(weight_decay=0.),
        'blocks.2.norm1.bias': dict(weight_decay=0.),
        'blocks.2.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.2.attn.proj.bias': dict(weight_decay=0.),
        'blocks.2.norm2.weight': dict(weight_decay=0.),
        'blocks.2.norm2.bias': dict(weight_decay=0.),
        'blocks.2.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.2.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.3.gamma_1': dict(weight_decay=0.),
        'blocks.3.gamma_2': dict(weight_decay=0.),
        'blocks.3.norm1.weight': dict(weight_decay=0.),
        'blocks.3.norm1.bias': dict(weight_decay=0.),
        'blocks.3.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.3.attn.proj.bias': dict(weight_decay=0.),
        'blocks.3.norm2.weight': dict(weight_decay=0.),
        'blocks.3.norm2.bias': dict(weight_decay=0.),
        'blocks.3.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.3.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.4.gamma_1': dict(weight_decay=0.),
        'blocks.4.gamma_2': dict(weight_decay=0.),
        'blocks.4.norm1.weight': dict(weight_decay=0.),
        'blocks.4.norm1.bias': dict(weight_decay=0.),
        'blocks.4.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.4.attn.proj.bias': dict(weight_decay=0.),
        'blocks.4.norm2.weight': dict(weight_decay=0.),
        'blocks.4.norm2.bias': dict(weight_decay=0.),
        'blocks.4.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.4.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.5.gamma_1': dict(weight_decay=0.),
        'blocks.5.gamma_2': dict(weight_decay=0.),
        'blocks.5.norm1.weight': dict(weight_decay=0.),
        'blocks.5.norm1.bias': dict(weight_decay=0.),
        'blocks.5.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.5.attn.proj.bias': dict(weight_decay=0.),
        'blocks.5.norm2.weight': dict(weight_decay=0.),
        'blocks.5.norm2.bias': dict(weight_decay=0.),
        'blocks.5.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.5.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.6.gamma_1': dict(weight_decay=0.),
        'blocks.6.gamma_2': dict(weight_decay=0.),
        'blocks.6.norm1.weight': dict(weight_decay=0.),
        'blocks.6.norm1.bias': dict(weight_decay=0.),
        'blocks.6.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.6.attn.proj.bias': dict(weight_decay=0.),
        'blocks.6.norm2.weight': dict(weight_decay=0.),
        'blocks.6.norm2.bias': dict(weight_decay=0.),
        'blocks.6.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.6.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.7.gamma_1': dict(weight_decay=0.),
        'blocks.7.gamma_2': dict(weight_decay=0.),
        'blocks.7.norm1.weight': dict(weight_decay=0.),
        'blocks.7.norm1.bias': dict(weight_decay=0.),
        'blocks.7.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.7.attn.proj.bias': dict(weight_decay=0.),
        'blocks.7.norm2.weight': dict(weight_decay=0.),
        'blocks.7.norm2.bias': dict(weight_decay=0.),
        'blocks.7.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.7.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.8.gamma_1': dict(weight_decay=0.),
        'blocks.8.gamma_2': dict(weight_decay=0.),
        'blocks.8.norm1.weight': dict(weight_decay=0.),
        'blocks.8.norm1.bias': dict(weight_decay=0.),
        'blocks.8.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.8.attn.proj.bias': dict(weight_decay=0.),
        'blocks.8.norm2.weight': dict(weight_decay=0.),
        'blocks.8.norm2.bias': dict(weight_decay=0.),
        'blocks.8.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.8.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.9.gamma_1': dict(weight_decay=0.),
        'blocks.9.gamma_2': dict(weight_decay=0.),
        'blocks.9.norm1.weight': dict(weight_decay=0.),
        'blocks.9.norm1.bias': dict(weight_decay=0.),
        'blocks.9.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.9.attn.proj.bias': dict(weight_decay=0.),
        'blocks.9.norm2.weight': dict(weight_decay=0.),
        'blocks.9.norm2.bias': dict(weight_decay=0.),
        'blocks.9.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.9.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.10.gamma_1': dict(weight_decay=0.),
        'blocks.10.gamma_2': dict(weight_decay=0.),
        'blocks.10.norm1.weight': dict(weight_decay=0.),
        'blocks.10.norm1.bias': dict(weight_decay=0.),
        'blocks.10.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.10.attn.proj.bias': dict(weight_decay=0.),
        'blocks.10.norm2.weight': dict(weight_decay=0.),
        'blocks.10.norm2.bias': dict(weight_decay=0.),
        'blocks.10.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.10.mlp.fc2.bias': dict(weight_decay=0.),

        'blocks.11.gamma_1': dict(weight_decay=0.),
        'blocks.11.gamma_2': dict(weight_decay=0.),
        'blocks.11.norm1.weight': dict(weight_decay=0.),
        'blocks.11.norm1.bias': dict(weight_decay=0.),
        'blocks.11.attn.qkv.bias': dict(weight_decay=0.),
        'blocks.11.attn.proj.bias': dict(weight_decay=0.),
        'blocks.11.norm2.weight': dict(weight_decay=0.),
        'blocks.11.norm2.bias': dict(weight_decay=0.),
        'blocks.11.mlp.fc1.bias': dict(weight_decay=0.),
        'blocks.11.mlp.fc2.bias': dict(weight_decay=0.),

        'norm.weight': dict(weight_decay=0.),
        'norm.bias': dict(weight_decay=0.),

        'head.bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None, update_interval=1)

lr_config = dict(
    policy='CosineAnnealingWarmupByEpoch',
    by_epoch=True,
    min_lr_ratio=0.00001/0.003,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.000001/0.003,
)
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 800

ema = dict(decay = 0.99996)
