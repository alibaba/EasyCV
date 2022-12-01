_base_ = './imagenet_timm_softmaxbased_jpg.py'

backbone_channels = 2048
feature_channels = 1536
num_classes = 300

metric_loss_name = 'ModelParallelSoftmaxLoss'
metric_loss_scale = 30
metric_loss_margin = 0.4

# model settings
model = dict(
    _delete_=True,
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
                    type=metric_loss_name,
                    loss_weight=1.0,
                    norm=False,
                    ddp=True,
                    scale=30,
                    margin=0.4,
                    embedding_size=feature_channels,
                    num_classes=num_classes)
            ],
            input_feature_index=[0])
    ])
