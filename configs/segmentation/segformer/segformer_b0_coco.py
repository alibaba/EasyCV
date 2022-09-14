# segformer of B0

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge',
    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
    'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter',
    'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence',
    'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
    'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
    'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
    'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow',
    'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
    'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
    'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other',
    'wood'
]
PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96],
           [128, 192, 192], [0, 64, 64], [0, 192, 224], [0, 192, 192],
           [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192],
           [0, 0, 224], [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
           [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
           [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
           [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0],
           [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0],
           [192, 128, 32], [128, 96, 128], [0, 0, 128], [64, 0, 32],
           [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
           [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128,
                                                            64], [192, 0, 32],
           [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64],
           [128, 128, 160], [64, 96, 0], [0, 128, 192], [0, 128, 160],
           [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
           [0, 64, 192], [0, 0, 32], [64, 160, 128], [128, 64, 64],
           [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
           [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0],
           [128, 64, 128], [64, 128, 96], [64, 160, 0], [0, 64, 0],
           [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224],
           [192, 160, 0], [0, 192, 0], [192, 128, 96], [192, 96, 128],
           [0, 64, 128], [64, 0, 96], [64, 224, 128], [128, 64, 0],
           [192, 0, 224], [64, 96, 128], [128, 192, 128], [64, 0, 224],
           [192, 224, 128], [128, 192, 64], [192, 0, 96], [192, 96, 0],
           [128, 64, 192], [0, 128, 96], [0, 224, 0], [64, 64, 64],
           [128, 128, 224], [0, 96, 0], [64, 192, 192], [0, 128, 224],
           [128, 224, 0], [64, 192, 64], [128, 128, 96], [128, 32, 128],
           [64, 0, 192], [0, 64, 96], [0, 160, 128], [192, 0, 64],
           [128, 64, 224], [0, 32, 128], [192, 128, 192], [0, 64, 224],
           [128, 160, 128], [192, 128, 0], [128, 64, 32], [128, 32, 64],
           [192, 0, 128], [64, 192, 32], [0, 160, 64], [64, 0, 0],
           [192, 192, 160], [0, 32, 64], [64, 128, 128], [64, 192, 160],
           [128, 160, 64], [64, 128, 0], [192, 192, 32], [128, 96, 192],
           [64, 0, 128], [64, 64, 32], [0, 224, 192], [192, 0, 0],
           [192, 64, 160], [0, 96, 192], [192, 128, 128], [64, 64, 160],
           [128, 224, 192], [192, 128, 64], [192, 64, 32], [128, 96, 64],
           [192, 0, 192], [0, 192, 32], [64, 224, 64], [64, 0, 64],
           [128, 192, 160], [64, 96, 64], [64, 128, 192], [0, 192, 160],
           [192, 224, 64], [64, 128, 64], [128, 192, 32], [192, 32, 192],
           [64, 64, 192], [0, 64, 32], [64, 160, 192], [192, 64, 64],
           [128, 64, 160], [64, 32, 192], [192, 192, 192], [0, 64, 160],
           [192, 160, 192], [192, 192, 0], [128, 64, 96], [192, 32, 64],
           [192, 64, 128], [64, 192, 96], [64, 160, 64], [64, 64, 0]]

num_classes = 172

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'SegDataset'
data_root = './data/coco_stuff164k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MMResize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='SegRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMPhotoMetricDistortion'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size=crop_size, pad_val=dict(img=0, masks=0, seg=255)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg')),
]
test_pipeline = [
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg')),
        ])
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ignore_index=255,
        data_source=dict(
            type='SegSourceRaw',
            img_suffix='.jpg',
            label_suffix='_labelTrainIds.png',
            img_root=data_root + 'train2017/',
            label_root=data_root + 'annotations/train2017/',
            split=data_root + 'train.txt',
            classes=CLASSES,
        ),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=1,
        ignore_index=255,
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_suffix='.jpg',
            label_suffix='_labelTrainIds.png',
            img_root=data_root + 'val2017/',
            label_root=data_root + 'annotations/val2017',
            split=data_root + 'val.txt',
            classes=CLASSES,
        ),
        pipeline=test_pipeline),
    test=dict(
        imgs_per_gpu=1,
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_suffix='.jpg',
            label_suffix='_labelTrainIds.png',
            img_root=data_root + 'val2017/',
            label_root=data_root + 'annotations/val2017',
            split=data_root + 'val.txt',
            classes=CLASSES,
        ),
        pipeline=test_pipeline))
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_options=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
total_epochs = 30
checkpoint_config = dict(interval=1)
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(
                type='SegmentationEvaluator',
                classes=CLASSES,
                metric_names=['mIoU'])
        ],
    )
]

predict = dict(type='SegmentationPredictor')

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')

cudnn_benchmark = False
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
