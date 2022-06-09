_base_ = 'configs/base.py'

# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='oss-cn-zhangjiakou.aliyuncs.com',  # your oss hosts
#     buckets=['your_bucket'])  # your oss buckets

work_dir = 'work_dirs/sop/swinb'

load_from = None

# model settings
model = dict(
    type='Classification',
    train_preprocess=['randomErasing'],
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_base_patch4_window7_224_22k_statedict.pth',
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='swin_base_patch4_window7_224_in22k'),
    neck=dict(
        type='RetrivalNeck',
        in_channels=1024,
        out_channels=1536,
        with_avg_pool=True,
        cdg_config=['G', 'S']),
    head=[
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels=1536,
            loss_config=[{
                'type': 'CrossEntropyLossWithLabelSmooth',
                'loss_weight': 1.0,
                'norm': True,
                'ddp': False,
                'label_smooth': 0.1,
                'temperature': 0.05,
                'with_cls': True,
                'embedding_size': 1536,
                'num_classes': 25000
            }],
            input_feature_index=[1]),
        dict(
            type='MpMetrixHead',
            with_avg_pool=True,
            in_channels=1536,
            loss_config=[{
                'type': 'CircleLoss',
                'loss_weight': 1.0,
                'norm': True,
                'm': 0.4,
                'gamma': 80,
                'ddp': True,
                'miner': {
                    'type': 'HDCMiner',
                    'filter_percentage': 0.4
                }
            }],
            input_feature_index=[0])
    ])

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data_train_list = 'dataset/metric_learning/sop_raw/meta/train_1.txt'
data_train_root = 'dataset/metric_learning/sop_raw'
data_test_list = 'dataset/metric_learning/sop_raw/meta/val_1.txt'
data_test_root = 'dataset/metric_learning/sop_raw'

data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            m_per_class=4,
            type='ClsSourceImageListByClass'),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type='ClsSourceImageList'),
        pipeline=test_pipeline),
)

eval_config = dict(interval=1, initial=False, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='extract',
        dist_eval=True,
        data=data['val'],
        evaluators=[
            dict(
                type='RetrivalTopKEvaluator',
                topk=(1, 2, 4, 8),
                metric_names=('R@K=1', 'R@K=8'))
        ],
    )
]

# additional hooks
custom_hooks = []

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='RMSprop', lr=8e-6, momentum=0.9, weight_decay=1e-4)
# optimizer = dict(type='RMSprop', lr=2e-5, momentum=0.9, weight_decay=1e-4)
# optimizer = dict(type='Adam', lr=1e-5, weight_decay=1e-4)
optimizer_config = dict()

# learning policy
# lr_config = dict(policy='step', step=[40, 70, 90])#warmup='linear')#, warmup_iters=500, warmup_ratio=0.001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.0001)  # warmup_by_epoch=True)

checkpoint_config = dict(interval=5)
# runtime settings
total_epochs = 100
