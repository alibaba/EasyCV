# model settings
model = dict(
    type='Detection',
    pretrained=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3, 4),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    head=dict(
        type='DINOHead',
        transformer=dict(
            type='DeformableTransformer',
            d_model=256,
            nhead=8,
            num_queries=900,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            num_patterns=0,
            modulate_hw_attn=True,
            # for deformable encoder
            deformable_encoder=True,
            deformable_decoder=True,
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            # init query
            decoder_query_perturber=None,
            add_channel_attention=False,
            random_refpoints_xy=False,
            # two stage
            two_stage_type=
            'standard',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
            two_stage_pat_embed=0,
            two_stage_add_query_num=0,
            two_stage_learn_wh=False,
            two_stage_keep_all_tokens=False,
            # evo of #anchors
            dec_layer_number=None,
            rm_dec_query_scale=True,
            rm_self_attn_layers=None,
            key_aware_type=None,
            # layer share
            layer_share_type=None,
            # for detach
            rm_detach=None,
            decoder_sa_type='sa',
            module_seq=['sa', 'ca', 'ffn'],
            # for dn
            embed_init_tgt=True,
            use_detached_boxes_dec_out=False),
        dn_components=dict(
            dn_number=100,
            dn_label_noise_ratio=0.5,  # paper 0.5, release code 0.25
            dn_box_noise_scale=1.0,
            dn_labelbook_size=80,
        ),
        num_classes=80,
        in_channels=[512, 1024, 2048],
        embed_dims=256,
        query_dim=4,
        num_queries=900,
        num_select=300,
        random_refpoints_xy=False,
        num_patterns=0,
        fix_refpoints_hw=-1,
        num_feature_levels=4,
        # two stage
        two_stage_type='standard',  # ['no', 'standard']
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        decoder_sa_type='sa',
        temperatureH=20,
        temperatureW=20,
        cost_dict=dict(
            cost_class=2,
            cost_bbox=5,
            cost_giou=2,
        ),
        weight_dict=dict(loss_ce=1, loss_bbox=5, loss_giou=2)))
