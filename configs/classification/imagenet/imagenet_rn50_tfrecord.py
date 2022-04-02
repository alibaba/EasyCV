_base_ = '../../base.py'
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    train_preprocess=['randomErasing'],
    # train_preprocess=['mixUp'],
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    # norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))

# dataset settings
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
size = 224
train_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=size, random_area=(0.08, 1.0)),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size, size],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
val_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliResize', resize_shorter=size * 1.15),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[size, size],
        mean=mean,
        std=std,
        prob=0.0)
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=4,
    train=dict(
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet-tfrecord/train-*',
            # root=data_root,  # pick one of `file_pattern` and `root&list_file`
            # list_file=data_root + 'meta/train.txt',
            cache_path='data/cache/train',
        ),
        pipeline=train_pipeline,
        label_offset=1),
    val=dict(
        imgs_per_gpu=50,
        type='DaliImageNetTFRecordDataSet',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet-tfrecord/train-*',
            # root=data_root,
            # list_file=data_root + 'meta/train_relative.txt',
            cache_path='data/cache/val',
        ),
        pipeline=val_pipeline,
        random_shuffle=False,
        label_offset=1))

eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
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
