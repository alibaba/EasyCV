_base_ = '../../base.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='swin_tiny_patch4_window7_224',
        num_classes=0,
    ),
    head=dict(type='ClsHead', in_channels=768, with_fc=True, num_classes=10))
# dataset settings
class_list = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]
data_source_cfg = dict(type='ClsSourceCifar10', root='data/cifar/')
dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
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
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]
custom_hooks = []

# optimizer
paramwise_options = {
    'norm': dict(weight_decay=0.),
    'bias': dict(weight_decay=0.),
    'absolute_pos_embed': dict(weight_decay=0.),
    'relative_position_bias_table': dict(weight_decay=0.)
}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
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

checkpoint_config = dict(interval=5)
# runtime settings
total_epochs = 5
# log setting
log_config = dict(interval=50)
# export config
export = dict(export_neck=True)
