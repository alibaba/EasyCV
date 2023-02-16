_base_ = './imagenet_resnet50_jpg.py'

checkpoint_sync_export = True
export = dict(export_neck=True)
class_list = None
total_epochs = 60
model = dict(
    neck=dict(
        type='ReIDNeck', in_channels=2048, out_channels=512, dropout=0.5),
    head=dict(in_channels=512, num_classes=751))
optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
    paramwise_options={'backbone': dict(lr_mult=0.1)})
lr_config = dict(step=[40], gamma=0.1)

data_source_type = 'ClsSourceImageList'
data_train_list = '/apsarapangu/disk2/yunji.cjy/Market1501/pytorch/meta/train_all.txt'
data_train_root = ''
data_test_list = '/apsarapangu/disk2/yunji.cjy/Market1501/pytorch/meta/val.txt'
data_test_root = ''
image_size2 = 224
image_size1 = int((256 / 224) * image_size2)

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_pipeline = [
    dict(type='Resize', size=image_size1),
    dict(type='CenterCrop', size=image_size2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

data = dict(
    train=dict(
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            class_list=class_list)),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type=data_source_type,
            class_list=class_list),
        pipeline=test_pipeline))

eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[
            dict(type='ClsEvaluator', topk=(1, 5), class_list=class_list)
        ],
    )
]
