_base_ = ['configs/ocr/recognition/rec_model_ch.py']

character_dict_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/dict/ta_dict.txt'
label_length = 128
model = dict(
    type='OCRRecNet',
    backbone=dict(
        type='OCRRecMobileNetV1Enhance',
        scale=0.5,
        last_conv_stride=[1, 2],
        last_pool_type='avg'),
    # inference
    # neck=dict(
    #     type='SequenceEncoder',
    #     in_channels=512,
    #     encoder_type='svtr',
    #     dims=64,
    #     depth=2,
    #     hidden_dims=120,
    #     use_guide=True),
    # head=dict(
    #     type='CTCHead',
    #     in_channels=64,
    #     fc_decay=0.00001),
    head=dict(
        type='MultiHead',
        in_channels=512,
        out_channels_list=dict(
            CTCLabelDecode=label_length + 2,
            SARLabelDecode=label_length + 4,
        ),
        head_list=[
            dict(
                type='CTCHead',
                Neck=dict(
                    type='svtr',
                    dims=64,
                    depth=2,
                    hidden_dims=120,
                    use_guide=True),
                Head=dict(fc_decay=0.00001, )),
            dict(type='SARHead', enc_dim=512, max_text_length=25)
        ]),
    postprocess=dict(
        type='CTCLabelDecode',
        character_dict_path=character_dict_path,
        use_space_char=True),
    loss=dict(
        type='MultiLoss',
        ignore_index=label_length + 3,
        loss_config_list=[
            dict(CTCLoss=None),
            dict(SARLoss=None),
        ]),
    pretrained=None)
