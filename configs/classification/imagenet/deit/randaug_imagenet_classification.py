data_train_list = 'data/imagenet1k/train.txt'
data_train_root = 'data/imagenet1k/train/'
data_test_list = 'data/imagenet1k/val.txt'
data_test_root = 'data/imagenet1k/val/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.08, 1.0),
        interpolation=3),  # interpolation='bicubic'
    dict(type='RandomHorizontalFlip'),
    dict(
        type='MMRandAugment',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x * 255) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='MMRandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[x * 255 for x in img_norm_cfg['mean'][::-1]],
        fill_std=[x * 255 for x in img_norm_cfg['std'][::-1]]),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=8,
    use_repeated_augment_sampler=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            type='ClsSourceImageList'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type='ClsSourceImageList'),
        pipeline=test_pipeline))

eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# used for unittest
predict = dict(
    type='ClassificationPredictor',
    pipelines=[
        dict(type='ToPILImage'),
        dict(type='Resize', size=256, interpolation=3),
        dict(type='CenterCrop', size=224),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Collect', keys=['img'])
    ])
