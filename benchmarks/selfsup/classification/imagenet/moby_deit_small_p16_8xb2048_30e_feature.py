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
# 1920: merge 4 layers of features, open models/backbones/vit_transfomer_dynamic.py:311: self.forward_return_n_last_blocks
# 384: default
feature_num = 1920
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(type='BenchMarkMLP', feature_num=feature_num),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=feature_num,
        num_classes=1000))
# dataset settings
data_source_cfg = dict(type='SSLSourceImageNetFeature')

root_path = 'linear_eval/imagenet_features/'
dataset_type = 'ClsDataset'
train_pipeline = []
test_pipeline = []

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
optimizer = dict(type='AdamW', lr=0.001, weight_decay=4e-5)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0, by_epoch=False)

checkpoint_config = dict(interval=5)
# runtime settings
total_epochs = 30
