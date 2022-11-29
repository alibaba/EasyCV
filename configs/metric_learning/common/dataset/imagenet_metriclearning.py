data_source_type = 'ClsSourceImageList'
data_train_list = '/apsarapangu/disk1/yunji.cjy/data/imagenet_raw/meta/train_labeled.txt'
data_train_root = '/apsarapangu/disk1/yunji.cjy/data/imagenet_raw/train/'
data_test_list = '/apsarapangu/disk1/yunji.cjy/data/imagenet_raw/meta/val_labeled.txt'
data_test_root = '/apsarapangu/disk1/yunji.cjy/data/imagenet_raw/validation/'
dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_size2 = 224
image_size1 = int((256 / 224) * image_size2)

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
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            type=data_source_type),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type=data_source_type),
        pipeline=test_pipeline))
