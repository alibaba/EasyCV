_base_ = 'common/dataset/imagenet_metriclearning.py'

backbone_channels = 2048
feature_channels = 1536
num_classes = 300

metric_loss_name = 'AMSoftmaxLoss'
metric_loss_scale = 30
metric_loss_margin = 0.4

# model settings
model = dict(
    type='Classification',
    backbone=dict(type='PytorchImageModelWrapper', model_name='resnet50'),
    neck=dict(
        type='RetrivalNeck',
        in_channels=backbone_channels,
        out_channels=feature_channels,
        with_avg_pool=True,
        cdg_config=['G', 'S']),
    head=[
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels=feature_channels,
            loss_config=[
                dict(
                    type='CrossEntropyLossWithLabelSmooth',
                    loss_weight=1.0,
                    norm=True,
                    ddp=False,
                    label_smooth=0.1,
                    temperature=0.05,
                    with_cls=True,
                    embedding_size=feature_channels,
                    num_classes=num_classes)
            ],
            input_feature_index=[1]),
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels=feature_channels,
            loss_config=[
                dict(
                    type=metric_loss_name,
                    loss_weight=1.0,
                    norm=False,
                    ddp=False,
                    scale=metric_loss_scale,
                    margin=metric_loss_margin,
                    embedding_size=feature_channels,
                    num_classes=num_classes)
            ],
            input_feature_index=[0])
    ])

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.0001)

checkpoint_config = dict(interval=5)
# runtime settings
total_epochs = 100

find_unused_parameters = True
