_base_ = '../../base.py'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2,4,4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(
        average_clips='prob',
        max_testing_views=4),
    pretrained='swin_tiny_patch4_window7_224_22k.pth')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='VideoResize', scale=(-1, 256)),
    dict(type='VideoRandomResizedCrop'),
    dict(type='VideoResize', scale=(224, 224), keep_ratio=False),
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
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='VideoResize', scale=(-1, 256)),
    dict(type='VideoCenterCrop', crop_size=224),
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
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='VideoResize', scale=(-1, 224)),
    dict(type='VideoThreeCrop', crop_size=224),
    dict(type='VideoFlip', flip_ratio=0),
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['imgs'])
]

data_root = '/home/yanhaiqiang.yhq/easycv_nfs/data/video/'
train_dataset = dict(
    type = 'VideoDataset',
    data_source = dict(
        type='VideoDatasource',
        ann_file = data_root+'kinetics400/test.txt',
        data_root = data_root,
        split = ' ', 
        ),
    pipeline=train_pipeline,
)

val_dataset = dict(
    type = 'VideoDataset',
    imgs_per_gpu=1,
    data_source = dict(
        type='VideoDatasource',
        ann_file = data_root+'kinetics400/test.txt',
        data_root = data_root,
        split = ' ', 
        ),
    pipeline=val_pipeline,
)

data = dict(
    imgs_per_gpu=8, workers_per_gpu=4, train=train_dataset, val=val_dataset)

# optimizer
total_epochs = 30
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.02,
    betas=(0.9, 0.999),
    paramwise_options={
        'backbone': dict(lr_mult=0.1),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
    })
optimizer_config = dict(update_interval=8)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)

checkpoint_config = dict(interval=1)

# eval 
eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]