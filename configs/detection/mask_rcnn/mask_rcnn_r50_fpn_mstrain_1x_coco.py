_base_ = [
    './_base_/models/mask_rcnn_r50_fpn.py', './_base_/datasets/coco.py',
    'configs/base.py'
]

cudnn_enabled = False

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

# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

checkpoint = '/apsarapangu/disk3/jiangnana.jnn/pretrained_models/selfsup/mmself_mocov2/backbone.pth'
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    # backbone=dict(
    #     frozen_stages=-1,
    #     norm_cfg=norm_cfg,
    #     norm_eval=False,
    #     init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

checkpoint_config = dict(interval=1)
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.0, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        # data=data['val'],
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
            dict(type='CocoMaskEvaluator', classes=CLASSES)
        ],
    )
]
load_from = '/home/jiangnana.jnn/workspace/EasyCV-ssh/EasyCV/work_dir/epoch_2.pth'
