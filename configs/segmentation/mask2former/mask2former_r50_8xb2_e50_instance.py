_base_ = ['configs/base.py']

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
    'hair drier', 'toothbrush'
]
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
           (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175),
           (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252),
           (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0),
           (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
           (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
           (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255),
           (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103),
           (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1),
           (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61),
           (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210),
           (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106),
           (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1),
           (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199),
           (0, 165, 120), (183, 130, 88), (95, 32, 0), (130, 114, 135),
           (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114),
           (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106),
           (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255),
           (201, 57, 1), (246, 0, 122), (191, 162, 208)]

model = dict(
    type='Mask2Former',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True),
    head=dict(
        type='Mask2FormerHead',
        pixel_decoder=dict(
            input_stride=[4, 8, 16, 32],
            input_channel=[256, 512, 1024, 2048],
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm='GN',
            transformer_in_features=[1, 2, 3],
            common_stride=4,
        ),
        transformer_decoder=dict(
            in_channels=256,
            num_classes=80,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
        ),
        num_things_classes=80,
        num_stuff_classes=0,
    ),
    train_cfg=dict(
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        deep_supervision=True,
        dec_layers=10,
        num_points=12554,
        no_object_weight=0.1,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    test_cfg=dict(
        instance_on=True,
        max_per_image=100,
    ),
    pretrained=True,
)
# dataset settings
data_root = 'database/coco/'
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(
        type='MMResize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='MMRandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='MMFilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='MMPad', size=image_size, pad_val=pad_cfg),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg')),
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMPad', size_divisor=32, pad_val=pad_cfg),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'ori_img_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg')),
        ])
]

train_dataset = dict(
    type='DetDataset',
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        # seg_prefix=data_root + 'panoptic_train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ],
        classes=CLASSES,
        filter_empty_gt=True,
        iscrowd=False,
    ),
    pipeline=train_pipeline)

val_dataset = dict(
    type='DetDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ],
        classes=CLASSES,
        test_mode=True,
        iscrowd=True,
    ),
    pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset, val=val_dataset)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_options={
        'backbone': dict(lr_mult=0.1),
        'query_embed': dict(weight_decay=0.),
        'query_feat': dict(weight_decay=0.),
        'level_embed': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
total_epochs = 50

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

checkpoint_config = dict(interval=1)

eval_config = dict(initial=False, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=True,
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
            dict(type='CocoMaskEvaluator', classes=CLASSES)
        ],
    )
]
