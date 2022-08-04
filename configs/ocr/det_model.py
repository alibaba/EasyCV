_base_ = ['configs/base.py']

model = dict(
    type='DBNet',
    backbone=dict(
        type='MobileNetV3',
        scale=0.5,
        model_name='large',
        disable_se=True),
    neck=dict(
        type='RSEFPN',
        in_channels=[16, 24, 56, 480],
        out_channels=96,
        shortcut=True),
    head=dict(
        type='DBHead',
        in_channels=96,
        k=50),
    postprocess=dict(
        type='DBPostProcess',
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode='fast'
    ),
    pretrained='/root/code/ocr/paddle_to_torch_tools/paddle_weights/ch_ptocr_v3_det_infer.pth'
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='MMResize', img_scale=(960,960)),
    dict(type='ResizeDivisor', size_divisor=32),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img','ori_shape']),
]