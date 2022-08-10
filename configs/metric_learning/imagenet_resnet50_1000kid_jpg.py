_base_ = 'configs/base.py'
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    # head=dict(
    #     type='ClsHead', with_avg_pool=True, in_channels=2048,
    #     num_classes=250000),
    head=dict(
        type='MpMetrixHead',
        with_avg_pool=True,
        in_channels=2048,
        loss_config=[
            # {
            #     "type": "ModelParallelSoftmaxLoss",
            #     "embedding_size": 2048,
            #     "num_classes" : 1000000,
            #     "norm" : False,
            #     'ddp': True,
            # }
            {
                'type': 'ModelParallelAMSoftmaxLoss',
                'embedding_size': 2048,
                'num_classes':
                1000,  # if CUDA out of memory, reduce num_classes.
                'norm': False,
                'ddp': True,
            }
        ],
        input_feature_index=[0]))

data_train_list = '/imagenet_raw/meta/train_labeled.txt'
data_train_root = '/imagenet_raw/train'
data_test_list = '/imagenet_raw/meta/val_labeled.txt'
data_test_root = '/imagenet_raw/val'
data_all_list = '/imagenet_raw/meta/all_labeled.txt'
data_root = '/imagenet_raw/'

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

eval_config = dict(interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='extract',
        dist_eval=True,
        data=data['val'],
        evaluators=[
            dict(
                type='RetrivalTopKEvaluator',
                topk=(1, 2, 4, 8),
                metric_names=('R@K=1', 'R@K=8'))
        ],
    )
]

# additional hooks
custom_hooks = []

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 90
