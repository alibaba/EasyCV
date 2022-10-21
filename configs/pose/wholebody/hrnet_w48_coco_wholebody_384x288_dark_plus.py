oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])

log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'  # noqa: E501
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

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
channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        arch='w48',
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
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=17))

train_pipeline = [
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3, unbiased_encoding=True),
    dict(
        type='PoseCollect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'image_id', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
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
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'

data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
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
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)
data_source_cfg = dict(type='WholeBodyCocoTopDownSource', data_cfg=data_cfg)

data = dict(
    imgs_per_gpu=32,  # for train
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='WholeBodyCocoTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_train_v1.0.json',
            img_prefix=f'{data_root}/train2017/',
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type='WholeBodyCocoTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=val_pipeline),
    test=dict(
        type='WholeBodyCocoTopDownDataset',
        data_source=dict(
            ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json',
            img_prefix=f'{data_root}/val2017/',
            test_mode=True,
            **data_source_cfg),
        pipeline=test_pipeline),
)

evaluation = dict(interval=10, metric=['mAP'], save_best='AP')

eval_config = dict(interval=10)
evaluator_args = dict(metric_names='AP')
eval_pipelines = [
    dict(
        mode='test',
        data=dict(**data['val'], imgs_per_gpu=1),
        evaluators=[dict(type='WholeBodyKeyPointEvaluator', **evaluator_args)])
]
export = dict(use_jit=False)
checkpoint_sync_export = True
predict = dict(type='WholeBodyKeypointsPredictor', bbox_thr=0.8)
