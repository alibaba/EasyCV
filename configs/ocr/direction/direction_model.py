_base_ = ['configs/base.py']

model = dict(
    type='TextClassifier',
    backbone=dict(type='OCRRecMobileNetV3', scale=0.35, model_name='small'),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=200,
        num_classes=2,
    ),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/cls/ch_ppocr_mobile_v2.0_cls/best_accuracy.pth'
)

train_pipeline = [
    dict(type='RecAug', use_tia=False),
    dict(type='ClsResizeImg', img_shape=(3, 48, 192)),
    dict(type='MMToTensor'),
    dict(type='Collect', keys=['img', 'label'], meta_keys=['img_path'])
]

val_pipeline = [
    dict(type='ClsResizeImg', img_shape=(3, 48, 192)),
    dict(type='MMToTensor'),
    dict(type='Collect', keys=['img', 'label'], meta_keys=['img_path'])
]

test_pipeline = [
    dict(type='ClsResizeImg', img_shape=(3, 48, 192)),
    dict(type='MMToTensor'),
    dict(type='Collect', keys=['img'], meta_keys=['img_path'])
]

train_dataset = dict(
    type='OCRClsDataset',
    data_source=dict(
        type='OCRClsSource',
        label_file='ocr/direction/pai/label_file/test_direction.txt',
        data_dir='ocr/direction/pai/img/test',
        label_list=['0', '180'],
    ),
    pipeline=train_pipeline)

val_dataset = dict(
    type='OCRClsDataset',
    data_source=dict(
        type='OCRClsSource',
        label_file='ocr/direction/pai/label_file/test_direction.txt',
        data_dir='ocr/direction/pai/img/test',
        label_list=['0', '180'],
        test_mode=True),
    pipeline=val_pipeline)

data = dict(
    imgs_per_gpu=512, workers_per_gpu=8, train=train_dataset, val=val_dataset)

total_epochs = 100
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

checkpoint_config = dict(interval=10)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

eval_config = dict(initial=True, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=False,
        evaluators=[dict(type='ClsEvaluator', topk=(1, ))],
    )
]
