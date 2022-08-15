# model settings
model = dict(
    type='Detection',
    pretrained='/root/pretrain/warpper_swin_large_patch4_window12_384_22k.pth',
    backbone=dict(
        type='DINOSwinTransformer',
        pretrain_img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        out_indices=(1, 2, 3),
        use_checkpoint=True),
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
            use_deformable_box_attn=False,
            box_attn_type='roi_align',
            # init query
            learnable_tgt_init=True,
            decoder_query_perturber=None,
            add_channel_attention=False,
            add_pos_value=False,
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
            rm_enc_query_scale=True,
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
            dn_type='cdn',
            dn_number=100,
            dn_label_noise_ratio=0.5,  # paper 0.5, release code 0.25
            dn_box_noise_scale=1.0,
            dn_labelbook_size=80,
        ),
        num_classes=80,
        in_channels=[384, 768, 1536],
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
