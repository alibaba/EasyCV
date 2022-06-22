_base_ = [
    './_base_/models/vitdet.py', './_base_/datasets/coco_instance.py',
    'configs/base.py'
]

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

model = dict(backbone=dict(aggregation='basicblock'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=10)
# optimizer
paramwise_options = {
    'norm': dict(weight_decay=0.),
    'bias': dict(weight_decay=0.),
    'pos_embed': dict(weight_decay=0.),
    'cls_token': dict(weight_decay=0.)
}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=None, loss_scale=512.)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.067,
    step=[88, 96])
total_epochs = 100

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
            dict(type='CocoMaskEvaluator', classes=CLASSES)
        ],
    )
]

find_unused_parameters = False
