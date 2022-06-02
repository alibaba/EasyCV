_base_ = 'configs/base.py'

# Studio2.0 config

# dataaug
img_size = 224

# channel setup
backbone_channel = 384
feature_channel = 1536
num_classes = 1000

# supported_backbone

# supported_softmax+loss ,all this loss use same parameters
# [
#    "ArcFaceLoss",    m=28.6, s=64
#    "CosFaceLoss",    m = 0.35, s=64
#    "LargeMarginSoftmaxLoss", m=4, s=1
#    "SphereFaceLoss",  m=4, s=1
#    "AMSoftmaxLoss"    m=0.4, s=30
#    "ModelParallelAMSoftmaxLoss" m=0.4, s=30
# ]

metric_loss_name = 'AMSoftmaxLoss'
metric_loss_weight = 1.0
metric_loss_scale = 30
metric_loss_margin = 0.4

evaluator1_name = 'ClsEvaluator'
evaluator2_name = 'RetrivalTopKEvaluator'

# ----------------------------- original config ------------------------------------#
log_config = dict(
    interval=1, hooks=[
        dict(type='TextLoggerHook'),
    ])

# should assign a dummy config dict  which will be re-assign by parametrize input
# oss io config
oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',
    buckets=['your_bucket'])

work_dir = 'oss://path/to/work_dirs/classification/'

# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='PytorchImageModelWrapper',

        # swin(dynamic)
        # model_name = 'dynamic_swin_tiny_p4_w7_224',
        # model_name = 'dynamic_swin_small_p4_w7_224',
        # model_name = 'dynamic_swin_base_p4_w7_224',

        # deit(224)
        # model_name='vit_deit_tiny_distilled_patch16_224',   # good 192,
        # model_name='vit_deit_small_distilled_patch16_224',   # good 192,

        # xcit(224)
        # model_name='xcit_small_12_p16',
        # model_name='xcit_medium_24_p16',
        # model_name='xcit_large_24_p8'

        # resnet
        model_name='resnet50',
        # model_name = 'resnet18',
        # model_name = 'resnet34',
        # model_name = 'resnet101',
        num_classes='${backbone_channel}',
    ),
    neck=dict(
        type='RetrivalNeck',
        in_channels='${backbone_channel}',
        out_channels='${feature_channel}',
        with_avg_pool=True,
        cdg_config=['G', 'S']),
    # neck=dict(
    #     type='FaceIDNeck',
    #     in_channels='${backbone_channel}',
    #     out_channels=${feature_channel}',),
    head=[
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels='${feature_channel}',
            loss_config=[
                dict(
                    type='CrossEntropyLossWithLabelSmooth',
                    loss_weight=1.0,
                    norm=True,
                    ddp=False,
                    label_smooth=0.1,
                    temperature=0.05,
                    with_cls=True,
                    embedding_size='${feature_channel}',
                    num_classes='${num_classes}')
            ],
            input_feature_index=[1]),
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels='${feature_channel}',
            loss_config=[
                dict(
                    type='${metric_loss_name}',
                    loss_weight='${metric_loss_weight}',
                    norm=False,
                    ddp=False,
                    scale='${metric_loss_scale}',
                    margin='${metric_loss_margin}',
                    embedding_size='${feature_channel}',
                    num_classes='${num_classes}')
            ],
            input_feature_index=[0])
    ])

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
train_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(
        type='DaliRandomResizedCrop',
        size='${img_size}',
        random_area=(0.2, 1.0)),
    dict(
        type='DaliCropMirrorNormalize',
        crop=['${img_size}', '${img_size}'],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
val_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliResize', resize_shorter='${img_size} * 1.15'),
    dict(
        type='DaliCropMirrorNormalize',
        crop=['${img_size}', '${img_size}'],
        mean=mean,
        std=std,
        prob=0.0)
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=1,
    train=dict(
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet_tfrecord/train-*',
            cache_path='data',
        ),
        pipeline=train_pipeline,
        label_offset=1),
    val=dict(
        imgs_per_gpu=50,
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet_tfrecord/validation-*',
            cache_path='data',
        ),
        pipeline=val_pipeline,
        random_shuffle=False,
        label_offset=1))

eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data='${data.val}',
        dist_eval=True,
        evaluators=[dict(type='${evaluator1_name}', topk=(1, 5))],
    ),
    dict(
        mode='extract',
        dist_eval=True,
        data='${data.val}',
        evaluators=[
            dict(
                type='${evaluator2_name}',
                topk=(1, 2),
                metric_names=('R@K=1', 'R@K=8'),
                feature_keyword=['neck', 'backbone'])
        ],
    )
]

# additional hooks
custom_hooks = []

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 100
load_from = None
resume_from = None

export = dict(export_neck=True)
