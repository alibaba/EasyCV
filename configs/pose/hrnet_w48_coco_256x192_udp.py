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

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
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

target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/'
    'EVTorch/modelzoo/pose/hrnet/hrnet_w48_pretrained.pth',
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
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_root = 'data/coco'

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file=
    f'{data_root}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='PoseCollect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'image_id', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='TopDownAffine', use_udp=True),
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
            'bbox_score', 'flip_pairs'
        ]),
]

test_pipeline = val_pipeline
data_source_cfg = dict(type='PoseTopDownSourceCoco', data_cfg=data_cfg)

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='PoseTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
            img_prefix=f'{data_root}/train2017/',
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type='PoseTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=val_pipeline),
    test=dict(
        type='PoseTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=test_pipeline),
)

eval_config = dict(interval=10, metric='mAP', save_best='AP')
evaluator_args = dict(soft_nms=False, use_nms=True, oks_thr=0.9, vis_thr=0.2)
eval_pipelines = [
    dict(
        mode='test',
        data=dict(**data['val'], imgs_per_gpu=1),
        evaluators=[dict(type='CoCoPoseTopDownEvaluator', **evaluator_args)])
]
export = dict(use_jit=False)
checkpoint_sync_export = True
