_base_ = '../imagenet/common/classification_base.py'

# oss_io_config = dict(ak_id='', # your oss ak id
#                      ak_secret='', # your oss ak secret
#                      hosts='', # your oss hosts
#                      buckets=[]) # your oss bucket name

# Ensure the CLASSES definition is in one line, for adapt to its replacement by user_config_params.
# yapf:disable
CLASSES = ['label1', 'label2', 'label3']  # replace with your true lables of itag manifest file
num_classes = 3
# model settings
model = dict(
    type='Classification',
    backbone=dict(type='MobileNetV2'),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=1280,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth',
            label_smooth=0,
        ),
        num_classes=num_classes))

train_itag_file = '/your/itag/train/file.manifest'  # or oss://your_bucket/data/train.manifest
test_itag_file = '/your/itag/test/file.manifest'  # oss://your_bucket/data/test.manifest

image_size2 = 224
image_size1 = int((256 / 224) * image_size2)
data_source_type = 'ClsSourceItag'
dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=image_size2),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=image_size1),
    dict(type='CenterCrop', size=image_size2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source_type,
            list_file=train_itag_file,
            class_list=CLASSES,
        ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source_type,
            list_file=test_itag_file,
            class_list=CLASSES),
        pipeline=test_pipeline))

eval_config = dict(initial=False, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, ))],
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=5)

# runtime settings
total_epochs = 100
checkpoint_sync_export = True
export = dict(export_neck=True)
