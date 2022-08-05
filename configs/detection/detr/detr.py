# model settings
model = dict(
    type='Detection',
    pretrained=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(4, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    head=dict(
        type='DETRHead',
        transformer=dict(
            type='DetrTransformer',
            in_channels=2048,
            num_queries=100,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
        ),
        num_classes=80,
        in_channels=2048,
        embed_dims=256,
        eos_coef=0.1,
        cost_dict=dict(
            cost_class=1,
            cost_bbox=5,
            cost_giou=2,
        ),
        weight_dict=dict(
            loss_ce=1,
            loss_bbox=5,
            loss_giou=2,
        )))
