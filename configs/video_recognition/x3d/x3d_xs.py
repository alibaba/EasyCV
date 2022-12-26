_base_ = '../../base.py'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='X3D',
        width_factor=2.0,
        depth_factor=2.2,
        bottlneck_factor=2.25,
        dim_c5=2048,
        dim_c1=12,
        num_classes=400,
        num_frames=4,
    ),
    # cls_head=dict(
    #     type='X3DHead',
    #     in_channels=192,
    #     num_classes=400,
    #     spatial_type='avg',
    #     dropout_ratio=0.5,
    #     fc1_bias=False),
    cls_head=dict(
        type='X3DHead',
        dim_in=192,
        dim_inner=432,
        dim_out=2048,
        num_classes=400,
        dropout_rate=0.5),
    test_cfg=dict(average_clips='prob'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=4, frame_interval=12, num_clips=1),
    dict(type='DecordDecode'),
    # dict(type='VideoResize', scale=(-1, 228)),
    dict(type='VideoRandomRescale', scale_range=(182, 228)),
    dict(type='VideoRandomResizedCrop'),
    dict(type='VideoResize', scale=(160, 160), keep_ratio=False),
    dict(type='VideoFlip', flip_ratio=0.5),
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=12,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='VideoResize', scale=(-1, 228)),
    dict(type='VideoCenterCrop', crop_size=160),
    dict(type='VideoFlip', flip_ratio=0),
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=12,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='VideoResize', scale=(-1, 182)),
    dict(type='VideoThreeCrop', crop_size=182),
    dict(type='VideoFlip', flip_ratio=0),
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['imgs'])
]

data_root = '/home/yanhaiqiang.yhq/easycv_nfs/data/video/'
train_dataset = dict(
    type='VideoDataset',
    data_source=dict(
        type='VideoDatasource',
        ann_file=data_root + 'kinetics400/test.txt',
        data_root=data_root,
        split=' ',
    ),
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='VideoDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='VideoDatasource',
        ann_file=data_root + 'kinetics400/test.txt',
        data_root=data_root,
        split=' ',
    ),
    pipeline=test_pipeline,
)

data = dict(
    imgs_per_gpu=128, workers_per_gpu=16, train=train_dataset, val=val_dataset)

# optimizer
total_epochs = 300
# optimizer = dict(
#     type='AdamW',
#     lr=0.1,
#     weight_decay=5e-5,
#     betas=(0.9, 0.999),
#     paramwise_options={
#         'bn': dict(weight_decay=0.),
#     })
# optimizer = dict(
#     type='SGD',
#     lr=0.1,
#     weight_decay=5e-5,
#     momentum=0.9,
#     nesterov=True,
#     paramwise_options={
#         'bn': dict(weight_decay=0.),
#     })
# optimizer = dict(
#     type='Adan',
#     lr=0.1,
#     weight_decay=5e-5,
#     paramwise_options={
#         'bn': dict(weight_decay=0.),
#     })
optimizer = dict(
    type='Adai',
    lr=0.8,
    weight_decay=5e-5,
    betas=(0.1, 0.99),
    decoupled=True,
    paramwise_options={
        'bn': dict(weight_decay=0.),
    })
# optimizer_config = dict(update_interval=8)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=35)

checkpoint_config = dict(interval=5)

# eval
eval_config = dict(initial=False, interval=5, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
