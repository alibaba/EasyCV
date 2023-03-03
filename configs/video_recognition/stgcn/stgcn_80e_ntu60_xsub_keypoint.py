_base_ = 'configs/base.py'

CLASSES = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up (from sitting position)',
    'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket',
    'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses',
    'take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up',
    'hand waving', 'kicking something', 'reach into pocket',
    'hopping (one foot jumping)', 'jump up', 'make a phone call/answer phone',
    'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something with finger', 'taking a selfie',
    'check time (from watch)', 'rub two hands together', 'nod head/bow',
    'shake head', 'wipe face', 'salute', 'put the palms together',
    'cross hands in front (say stop)', 'sneeze/cough', 'staggering', 'falling',
    'touch head (headache)', 'touch chest (stomachache/heart pain)',
    'touch back (backache)', 'touch neck (neckache)',
    'nausea or vomiting condition',
    'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person', 'kicking other person',
    'pushing other person', 'pat on back of other person',
    'point finger at the other person', 'hugging other person',
    'giving something to other person', "touch other person's pocket",
    'handshaking', 'walking towards each other',
    'walking apart from each other'
]

model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'VideoDataset'
ann_file_train = 'data/posec3d/ntu60_xsub_train.pkl'
ann_file_val = 'data/posec3d/ntu60_xsub_val.pkl'

train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='VideoToTensor', keys=['keypoint'])
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='PoseDataSourceForVideoRec',
            ann_file=ann_file_train,
            data_prefix='',
        ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        imgs_per_gpu=1,
        data_source=dict(
            type='PoseDataSourceForVideoRec',
            ann_file=ann_file_val,
            data_prefix='',
        ),
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type='PoseDataSourceForVideoRec',
            ann_file=ann_file_val,
            data_prefix='',
        ),
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 80

# eval
eval_config = dict(initial=False, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=1)

export = dict(type='raw')
# export = dict(type='jit')
# export = dict(
#     type='blade',
#     blade_config=dict(
#         enable_fp16=True,
#         fp16_fallback_op_ratio=0.0,
#         customize_op_black_list=[
#             'aten::select', 'aten::index', 'aten::slice', 'aten::view',
#             'aten::upsample', 'aten::clamp', 'aten::clone'
#         ]))
