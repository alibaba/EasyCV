_base_ = ['configs/base.py']

character_dict_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/dict/ppocr_keys_v1.txt'

model = dict(
    type='OCRRecNet',
    backbone=dict(
        type='OCRRecMobileNetV1Enhance',
        scale=0.5,
        last_conv_stride=[1, 2],
        last_pool_type='avg'),
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
            CTCLabelDecode=6625,
            SARLabelDecode=6627,
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
        ignore_index=6626,
        loss_config_list=[
            dict(CTCLoss=None),
            dict(SARLoss=None),
        ]),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/ch_PP-OCRv3_rec/best_accuracy_student.pth'
)

train_pipeline = [
    dict(type='RecConAug', prob=0.5, image_shape=(48, 320, 3)),
    dict(type='RecAug'),
    dict(
        type='MultiLabelEncode',
        max_text_length=25,
        use_space_char=True,
        character_dict_path=character_dict_path,
    ),
    dict(type='RecResizeImg', image_shape=(3, 48, 320)),
    dict(type='MMToTensor'),
    dict(
        type='Collect',
        keys=['img', 'label_ctc', 'label_sar', 'length', 'valid_ratio'],
        meta_keys=['img_path'])
]

val_pipeline = [
    dict(
        type='MultiLabelEncode',
        max_text_length=25,
        use_space_char=True,
        character_dict_path=character_dict_path,
    ),
    dict(type='RecResizeImg', image_shape=(3, 48, 320)),
    dict(type='MMToTensor'),
    dict(
        type='Collect',
        keys=['img', 'label_ctc', 'label_sar', 'length', 'valid_ratio'],
        meta_keys=['img_path'])
]

test_pipeline = [
    dict(type='RecResizeImg', image_shape=(3, 48, 320)),
    dict(type='MMToTensor'),
    dict(type='Collect', keys=['img'], meta_keys=['img_path'])
]

train_dataset = dict(
    type='OCRRecDataset',
    data_source=dict(
        type='OCRRecSource',
        label_file='ocr/rec/pai/label_file/train.txt',
        data_dir='ocr/rec/pai/img/train',
        ext_data_num=2,
    ),
    pipeline=train_pipeline)

val_dataset = dict(
    type='OCRRecDataset',
    data_source=dict(
        type='OCRRecSource',
        label_file='ocr/rec/pai/label_file/test.txt',
        data_dir='ocr/rec/pai/img/test',
        ext_data_num=0,
    ),
    pipeline=val_pipeline)

data = dict(
    imgs_per_gpu=128, workers_per_gpu=4, train=train_dataset, val=val_dataset)

total_epochs = 10
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

eval_config = dict(initial=True, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=False,
        evaluators=[dict(type='OCRRecEvaluator', ignore_space=False)],
    )
]
