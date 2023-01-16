_base_ = '../../base.py'
num_classes = 1206
multi_class = True
model = dict(
    type='ClipBertTwoStream',
    vision=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    text=dict(
        type='ClipBertClassification',
        config_text=dict(
            backbone_channel_in_size=2048,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=10,
            pad_token_id=1,
            type_vocab_size=2,
            vocab_size=21128,
        ),
        config_cross=dict(
            backbone_channel_in_size=2048,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=2,
            pad_token_id=1,
            num_labels=num_classes,
        ),
    ),
    vison_pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/video/backbone/swin_tiny_patch4_window7_224_22k.pth',
    text_pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/video/bert-base-chinese/pytorch_model.bin',
    multi_class=multi_class)

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
    dict(type='TextTokenizer'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'label', 'text_input_ids', 'text_input_mask'],
        meta_keys=[]),
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
    dict(type='VideoResize', scale=(-1, 224)),
    dict(type='VideoCenterCrop', crop_size=224),
    dict(type='VideoFlip', flip_ratio=0),
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(type='TextTokenizer'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'label', 'text_input_ids', 'text_input_mask'],
        meta_keys=['filename']),
    dict(type='VideoToTensor', keys=['imgs', 'label'])
]

data_root = 'data/video/video_text_multilabel/'
train_ann_file = 'data/video/video_text_multilabel/test.txt'
val_ann_file = 'video/video_text_multilabel/test.txt'
train_dataset = dict(
    type='VideoDataset',
    data_source=dict(
        type='VideoTextDatasource',
        ann_file=train_ann_file,
        data_root=data_root,
        multi_class=multi_class,
        num_classes=num_classes),
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='VideoDataset',
    data_source=dict(
        type='VideoTextDatasource',
        ann_file=val_ann_file,
        data_root=data_root,
        multi_class=multi_class,
        num_classes=num_classes),
    pipeline=val_pipeline,
)

data = dict(
    imgs_per_gpu=2, workers_per_gpu=4, train=train_dataset, val=val_dataset)

total_epochs = 30
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_options={
        'bert.encoder_text': dict(lr_mult=0.1),
        'bert.embeddings': dict(lr_mult=0.1),
    })

optimizer_config = dict(update_interval=8)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2)

checkpoint_config = dict(interval=1)

# eval
eval_config = dict(initial=True, interval=1, gpu_collect=True)

evaluators_type = 'MultiLabelEvaluator'
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type=evaluators_type)],
    )
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
