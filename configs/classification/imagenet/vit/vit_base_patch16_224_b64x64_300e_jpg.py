_base_ = '../common/dataset/imagenet_classification.py'

num_classes = 1000
# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    mixup_cfg=dict(
        mixup_alpha=0.2,
        prob=1.0,
        mode='batch',
        label_smoothing=0.1,
        num_classes=num_classes),
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='vit_base_patch16_224',
        num_classes=num_classes,
    ),
    head=dict(
        type='ClsHead',
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=False))

image_size2 = 224
image_size1 = int((256 / 224) * image_size2)
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=image_size2),
    dict(type='RandomHorizontalFlip'),
    dict(type='MMAutoAugment'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=image_size1),
    dict(type='CenterCrop', size=image_size2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

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
total_epochs = 300
