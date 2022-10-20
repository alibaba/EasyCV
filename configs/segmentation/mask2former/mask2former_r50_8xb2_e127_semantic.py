_base_ = ['configs/base.py']

CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route',
    'bed', 'window ', 'grass', 'cabinet', 'sidewalk, pavement', 'person',
    'earth, ground', 'door', 'table', 'mountain, mount', 'plant', 'curtain',
    'chair', 'car', 'water', 'painting, picture', 'sofa', 'shelf', 'house',
    'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk',
    'rock, stone', 'wardrobe, closet, press', 'lamp', 'tub', 'rail', 'cushion',
    'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign',
    'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator, icebox',
    'grandstand, covered stand', 'path', 'stairs', 'runway',
    'case, display case, showcase, vitrine',
    'pool table, billiard table, snooker table', 'pillow',
    'screen door, screen', 'stairway, staircase', 'river', 'bridge, span',
    'bookcase', 'blind, screen', 'coffee table',
    'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower',
    'book', 'hill', 'bench', 'countertop', 'stove', 'palm, palm tree',
    'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel, hut, hutch, shack, shanty', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning, sunshade, sunblind',
    'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole',
    'land, ground, soil',
    'bannister, banister, balustrade, balusters, handrail',
    'escalator, moving staircase, moving stairway',
    'ottoman, pouf, pouffe, puff, hassock', 'bottle',
    'buffet, counter, sideboard',
    'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship',
    'fountain',
    'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy',
    'washer, automatic washer, washing machine', 'plaything, toy', 'pool',
    'stool', 'barrel, cask', 'basket, handbasket', 'falls', 'tent', 'bag',
    'minibike, motorbike', 'cradle', 'oven', 'ball', 'food, solid food',
    'step, stair', 'tank, storage tank', 'trade name', 'microwave', 'pot',
    'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket, cover',
    'sculpture', 'hood, exhaust hood', 'sconce', 'vase', 'traffic light',
    'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
    'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock',
    'flag'
]

PALETTE = [(120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
           (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
           (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
           (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
           (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
           (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
           (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
           (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
           (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
           (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
           (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
           (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
           (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
           (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
           (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
           (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
           (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
           (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
           (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
           (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
           (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
           (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
           (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
           (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
           (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
           (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
           (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
           (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
           (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
           (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
           (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
           (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
           (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
           (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
           (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
           (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
           (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
           (102, 255, 0), (92, 0, 255)]

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
            num_classes=150,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
        ),
        num_things_classes=150,
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
        semantic_on=True,
        max_per_image=100,
    ),
    pretrained=True,
)

data_root = 'ADEChallengeData2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MMResize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='SegRandomCrop', crop_size=crop_size),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMPhotoMetricDistortion'),
    dict(type='MMPad', size=crop_size),
    dict(type='MMNormalize', **img_norm_cfg),
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

train_dataset = dict(
    type='SegDataset',
    data_source=dict(
        type='SegSourceRaw',
        cache_on_the_fly=True,
        img_root=data_root + 'images/training',
        label_root=data_root + 'annotations/training',
        reduce_zero_label=True,
        classes=CLASSES,
    ),
    pipeline=train_pipeline)

val_dataset = dict(
    type='SegDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='SegSourceRaw',
        cache_on_the_fly=True,
        img_root=data_root + 'images/validation',
        label_root=data_root + 'annotations/validation',
        reduce_zero_label=True,
        classes=CLASSES,
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
# it seems grad clip not influence result
# optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
total_epochs = 127

lr_config = dict(
    policy='Poly',
    min_lr=0,
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1e-4,
    warmup_by_epoch=False,
    by_epoch=False,
    power=0.9)
checkpoint_config = dict(interval=1)

eval_config = dict(initial=False, interval=1, gpu_collect=False)

eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(
                type='SegmentationEvaluator',
                classes=CLASSES,
                ignore_index=255,
                metric_names=['mIoU'])
        ],
    )
]
