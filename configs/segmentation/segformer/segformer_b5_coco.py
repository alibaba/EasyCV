_base_ = './segformer_b0_coco.py'

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

model = dict(
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth',
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=768,
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (2048, 640)
crop_size = (640, 640)
train_pipeline = [
    dict(type='MMResize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
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
        img_scale=img_scale,
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

data_root = './data/coco_stuff164k/'
# dataset settings
data_type = 'SegSourceRaw'
data_root = 'data/VOCdevkit/VOC2012'

train_img_root = data_root + 'JPEGImages'
train_label_root = data_root + 'SegmentationClass'
train_list_file = data_root + 'ImageSets/Segmentation/train.txt'

val_img_root = data_root + 'JPEGImages'
val_label_root = data_root + 'SegmentationClass'
val_list_file = data_root + 'ImageSets/Segmentation/val.txt'

test_batch_size = 2

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='SegDataset',
        ignore_index=255,
        data_source=dict(
            type=data_type,
            img_suffix='.jpg',
            label_suffix='.png',
            img_root=train_img_root,
            label_root=train_label_root,
            split=train_list_file,
            classes=CLASSES,
        ),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=test_batch_size,
        ignore_index=255,
        type='SegDataset',
        data_source=dict(
            type=data_type,
            img_suffix='.jpg',
            label_suffix='.png',
            img_root=val_img_root,
            label_root=val_label_root,
            split=val_list_file,
            classes=CLASSES,
        ),
        pipeline=test_pipeline),
)
