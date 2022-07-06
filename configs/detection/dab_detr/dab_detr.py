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
        type='DABDETRHead',
        transformer=dict(
            type='DABDetrTransformer',
            in_channels=2048,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            activation='prelu',
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            keep_query_pos=False,
            query_scale_type='cond_elewise',
            modulate_hw_attn=True,
            bbox_embed_diff_each_layer=False,
            temperatureH=20,
            temperatureW=20),
        num_classes=80,
        in_channels=2048,
        embed_dims=256,
        query_dim=4,
        iter_update=True,
        num_queries=300,
        num_select=300,
        random_refpoints_xy=False,
        num_patterns=0,
        bbox_embed_diff_each_layer=False,
        cost_dict=dict(
            cost_class=2,
            cost_bbox=5,
            cost_giou=2,
        ),
        weight_dict=dict(
            loss_ce=1,
            loss_bbox=5,
            loss_giou=2,
        )))
