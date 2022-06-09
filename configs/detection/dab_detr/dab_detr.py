# model settings
model = dict(
    type='DETR',
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet50.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(4, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='DABDetrTransformer',
        in_channels=2048,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='prelu',
        normalize_before=False,
        return_intermediate_dec=True,
    ),
    head=dict(
        type='DABDETRHead',
        num_classes=80,
        in_channels=2048,
        embed_dims=256,
        use_sigmoid=False,
        cost_dict={
            'cost_class': 2,
            'cost_bbox': 5,
            'cost_giou': 2,
        },
        weight_dict={
            'loss_ce': 1,
            'loss_bbox': 5,
            'loss_giou': 2
        },
    ))
