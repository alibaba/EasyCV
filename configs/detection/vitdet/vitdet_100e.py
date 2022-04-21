_base_ = [
    './_base_/models/vitdet.py',
    './_base_/datasets/coco.py',
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

checkpoint_config = dict(interval=10)
# optimizer
optimizer = dict(
    type='_AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_options=dict(
        num_layers=12, 
        layer_decay_rate=0.7,
        custom_keys={
            'bias': dict(decay_multi=0.),
            'pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            "rel_pos_h": dict(decay_mult=0.),
            "rel_pos_w": dict(decay_mult=0.),
            }
        )
    )
optimizer_config = dict(grad_clip=None)
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

load_from = '/home/yunji.cjy/pretrain/mae_pretrain_vit_base_1600e.pth'