_base_ = ['configs/base.py']

model = dict(
    type='DBNet',
    backbone=dict(
        type='OCRDetMobileNetV3',
        scale=0.5,
        model_name='large',
        disable_se=True),
    neck=dict(
        type='RSEFPN',
        in_channels=[16, 24, 56, 480],
        out_channels=96,
        shortcut=True),
    head=dict(type='DBHead', in_channels=96, k=50),
    postprocess=dict(
        type='DBPostProcess',
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode='fast'),
    loss=dict(
        type='DBLoss',
        balance_loss=True,
        main_loss_type='DiceLoss',
        alpha=5,
        beta=10,
        ohem_ratio=3),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/en_PP-OCRv3_det/student.pth'
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(
        type='IaaAugment',
        augmenter_args=[{
            'type': 'Fliplr',
            'args': {
                'p': 0.5
            }
        }, {
            'type': 'Affine',
            'args': {
                'rotate': [-10, 10]
            }
        }, {
            'type': 'Resize',
            'args': {
                'size': [0.5, 3]
            }
        }]),
    dict(
        type='EastRandomCropData',
        size=[640, 640],
        max_tries=50,
        keep_ratio=True),
    dict(
        type='MakeBorderMap', shrink_ratio=0.4, thresh_min=0.3,
        thresh_max=0.7),
    dict(type='MakeShrinkMap', shrink_ratio=0.4, min_text_size=8),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(
        type='ImageToTensor',
        keys=[
            'img', 'threshold_map', 'threshold_mask', 'shrink_map',
            'shrink_mask'
        ]),
    dict(
        type='Collect',
        keys=[
            'img', 'threshold_map', 'threshold_mask', 'shrink_map',
            'shrink_mask'
        ]),
]

# test_pipeline = [
#     dict(type='MMResize', img_scale=(960, 960)),
#     dict(type='ResizeDivisor', size_divisor=32),
#     dict(type='MMNormalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(
#         type='Collect',
#         keys=['img'],
#         meta_keys=['ori_img_shape', 'polys', 'ignore_tags']),
# ]
test_pipeline = [
    dict(type='OCRDetResize', limit_side_len=640, limit_type='min'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['ori_img_shape', 'polys', 'ignore_tags']),
]

val_pipeline = [
    dict(type='OCRDetResize', image_shape=(736, 1280)),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['ori_img_shape', 'polys', 'ignore_tags']),
]

data_root = '/apsarapangu/disk1/yunji.cjy/data/icdar2015/'
train_ann_file = '/apsarapangu/disk1/yunji.cjy/data/icdar2015/train_icdar2015_label.txt'
val_ann_file = '/apsarapangu/disk1/yunji.cjy/data/icdar2015/test_icdar2015_label.txt'

train_dataset = dict(
    type='OCRDetDataset',
    data_source=dict(
        type='OCRDetSource', label_file=train_ann_file, data_dir=data_root),
    pipeline=train_pipeline)

val_dataset = dict(
    type='OCRDetDataset',
    imgs_per_gpu=2,
    data_source=dict(
        type='OCRDetSource',
        label_file=val_ann_file,
        data_dir=data_root,
        test_mode=True),
    pipeline=val_pipeline)

data = dict(
    imgs_per_gpu=16, workers_per_gpu=2, train=train_dataset, val=val_dataset)

total_epochs = 100
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999))

# learning policy
lr_config = dict(policy='fixed')

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

eval_config = dict(initial=False, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=True,
        evaluators=[dict(type='OCRDetEvaluator')],
    )
]
