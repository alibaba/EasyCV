_base_ = ['configs/base.py']

model = dict(
    type='DBNet',
    backbone=dict(
        type='MobileNetV3', scale=0.5, model_name='large', disable_se=True),
    neck=dict(
        type='RSEFPN',
        in_channels=[16, 24, 56, 480],
        out_channels=96,
        # out_channels=256,
        shortcut=True),
    head=dict(
        type='DBHead',
        in_channels=96,
        # in_channels=256,
        k=50),
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
    # pretrained='/mnt/workspace/code/ocr/paddle_to_torch_tools/paddle_weights/ch_ptocr_v3_det_infer.pth'
    # pretrained='/mnt/data/code/ocr/PaddleOCR/pretrain_models/MobileNetV3_large_x0_5_pretrained.pth'
    pretrained=
    '/root/code/ocr/PaddleOCR/pretrain_models/Multilingual_PP-OCRv3_det_distill_train/student.pth'
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
#     dict(type='MMResize', img_scale=(960,960)),
#     dict(type='ResizeDivisor', size_divisor=32),
#     dict(type='MMNormalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'],meta_keys=['ori_img_shape','polys','ignore_tags']),
# ]

test_pipeline = [
    dict(type='DetResizeForTest', limit_side_len=640, limit_type='min'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['ori_img_shape', 'polys', 'ignore_tags']),
]

val_pipeline = [
    dict(type='DetResizeForTest', limit_side_len=640, limit_type='min'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['ori_img_shape', 'polys', 'ignore_tags']),
]

train_dataset = dict(
    type='OCRDetDataset',
    data_source=dict(
        type='OCRPaiDetSource',
        label_file=[
            '/nas/database/ocr/det/pai/label_file/train/20191218131226_npx_e2e_train.csv',
            '/nas/database/ocr/det/pai/label_file/train/20191218131302_social_e2e_train.csv',
            '/nas/database/ocr/det/pai/label_file/train/20191218233728_synth_e2e_english_train_2000.csv',
            '/nas/database/ocr/det/pai/label_file/train/20191218122330_book_e2e_train.csv',
        ],
        data_dir='/nas/database/ocr/det/pai/img/train'),
    pipeline=train_pipeline)

val_dataset = dict(
    type='OCRDetDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='OCRPaiDetSource',
        label_file=[
            '/nas/database/ocr/det/pai/label_file/test/20191218131744_npx_e2e_test.csv',
            '/nas/database/ocr/det/pai/label_file/test/20191218131817_social_e2e_test.csv'
        ],
        data_dir='/nas/database/ocr/det/pai/img/test'),
    pipeline=val_pipeline)

data = dict(
    imgs_per_gpu=16, workers_per_gpu=2, train=train_dataset, val=val_dataset)

total_epochs = 100
optimizer = dict(
    type='Adam',
    # lr=0.001,
    lr=0.001,
    betas=(0.9, 0.999))

# learning policy
# lr_config = dict(policy='fixed')

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
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

eval_config = dict(initial=True, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=True,
        evaluators=[dict(type='OCRDetEvaluator')],
    )
]
