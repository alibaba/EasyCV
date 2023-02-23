_base_ = './fcos_r50_torch_1x_coco.py'

data_root0 = 'data/tracking/crowdhuman/'
data_root1 = 'data/tracking/MOT20/'
CLASSES = ('pedestrian', )
train_dataset = dict(
    data_source=dict(
        ann_file=[
            data_root1 + 'annotations/train_cocoformat.json', data_root0 +
            '/annotations/crowdhuman_train.json', data_root0 +
            '/annotations/crowdhuman_val.json'
        ],
        img_prefix=[
            data_root1 + 'train', data_root0 + 'train', data_root0 + 'val'
        ],
        classes=CLASSES))

val_dataset = dict(
    data_source=dict(
        ann_file=data_root0 + '/annotations/crowdhuman_val.json',
        img_prefix=data_root0 + 'val',
        classes=CLASSES))

data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset, val=val_dataset)

model = dict(head=dict(num_classes=1))

optimizer = dict(lr=0.001)

eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]

checkpoint_config = dict(interval=1)
checkpoint_sync_export = True
export = dict(export_neck=True)

load_from = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/fcos_epoch_12.pth'
