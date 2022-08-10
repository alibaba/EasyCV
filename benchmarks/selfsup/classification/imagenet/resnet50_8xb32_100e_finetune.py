_base_ = 'configs/base.py'

# oss config only works when using oss
# sync local models and logs to oss
oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])
oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='your oss hosts',
    buckets=['your oss buckets'])

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))

base_root = 'data/imagenet_raw/'
data_train_list = base_root + 'meta/train_labeled.txt'
data_train_root = base_root + 'train'
data_test_list = base_root + 'meta/val_labeled.txt'
data_test_root = base_root + 'val'
dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='ClsSourceImageList',
            list_file=data_train_list,
            root=data_train_root),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type='ClsSourceImageList',
            list_file=data_test_list,
            root=data_test_root),
        pipeline=test_pipeline))

eval_config = dict(interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))])
]

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)

# learning policy
lr_config = dict(policy='step', step=[60, 80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
load_from = None
