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
    backbone=dict(type='BenchMarkMLP', feature_num=2048),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))
# dataset settings
data_source_cfg = dict(type='SSLSourceImageNetFeature')

root_path = 'linear_eval/imagenet_features/'
dataset_type = 'ClsDataset'
train_pipeline = [
    # dict(type='ToTensor'),
]
test_pipeline = [
    # dict(type='ToTensor'),
]

data = dict(
    imgs_per_gpu=2048,  # total 2048*8=256, 8GPU linear cls
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            root_path=root_path, training=True, **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            root_path=root_path, training=False, **data_source_cfg),
        pipeline=test_pipeline))

# additional hooks

eval_config = dict(interval=5, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))])
]

# optimizer
optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[20, 30])

# runtime settings
total_epochs = 40

checkpoint_config = dict(interval=5)
