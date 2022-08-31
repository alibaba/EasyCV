# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='oss-cn-zhangjiakou.aliyuncs.com',  # your oss hosts
#     buckets=['your_bucket'])  # your oss buckets

oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)

optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=False,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False))),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=11))

data_root = 'data/coco'

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
)

train_pipeline = [
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='PoseCollect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'image_id', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'flip_pairs'
        ])
]

val_pipeline = [
    dict(type='TopDownAffine'),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='PoseCollect',
        keys=['img'],
        meta_keys=[
            'image_file', 'image_id', 'center', 'scale', 'rotation',
            'flip_pairs'
        ])
]

test_pipeline = val_pipeline
data_source_cfg = dict(type='HandCocoPoseTopDownSource', data_cfg=data_cfg)

data = dict(
    imgs_per_gpu=32,  # for train
    workers_per_gpu=2,  # for train
    # imgs_per_gpu=1,  # for test
    # workers_per_gpu=1,  # for test
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='HandCocoWholeBodyDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_train_v1.0.json',
            img_prefix=f'{data_root}/train2017/',
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type='HandCocoWholeBodyDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=val_pipeline),
    test=dict(
        type='HandCocoWholeBodyDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=val_pipeline),
)

eval_config = dict(interval=10, metric='PCK', save_best='PCK')
evaluator_args = dict(
    metric_names=['PCK', 'AUC', 'EPE', 'NME'], pck_thr=0.2, auc_nor=30)
eval_pipelines = [
    dict(
        mode='test',
        data=dict(**data['val'], imgs_per_gpu=1),
        evaluators=[dict(type='KeyPointEvaluator', **evaluator_args)])
]
export = dict(use_jit=False)
checkpoint_sync_export = True
predict = dict(type='HandKeypointsPredictor')
