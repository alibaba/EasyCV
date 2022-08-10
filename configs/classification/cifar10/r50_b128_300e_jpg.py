_base_ = '../../base.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 4: stage-4
        norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=10))
# dataset settings
class_list = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]
data_source_cfg = dict(type='ClsSourceCifar10', root='data/cifar/')
dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
data = dict(
    imgs_per_gpu=128,
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
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='step', step=[150, 250])
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 350
# log setting
log_config = dict(interval=100)
# export config
export = dict(export_neck=True)
