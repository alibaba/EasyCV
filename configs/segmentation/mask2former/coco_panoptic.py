data_root = "/home/yanhaiqiang.yhq/database/coco/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(
    #     type='LoadPanopticAnnotations',
    #     with_bbox=True,
    #     with_mask=True,
    #     with_seg=True),
    dict(type='MMResize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=32),
    # dict(type='MMSegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

train_dataset = dict(
    type='DetDataset',
    data_source=dict(
        type='DetSourceCocoPanoptic',
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'panoptic_val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadPanopticAnnotations', with_bbox=True, with_mask=True, with_seg=True)
        ],
        # classes=CLASSES,
        filter_empty_gt=True,
        iscrowd=False,
    ),
    pipeline=train_pipeline)


data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset)