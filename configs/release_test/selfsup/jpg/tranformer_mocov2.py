_base_ = 'configs/base.py'
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    train_preprocess=['gaussianBlur'],
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',
        model_name='swin_small_patch4_window7_224',
        # model_name='swin_tiny_patch4_window7_224',
        # model_name='swin_base_patch4_window7_224_in22k',
        # model_name='vit_deit_tiny_distilled_patch16_224',
        # model_name = 'vit_deit_small_distilled_patch16_224',
        # model_name = 'resnet50',
        # num_classes=1000,
        pretrained=True,
    ),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=768,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(type='ClsSourceImageList', )

data_train_list = 'data/imagenet_raw/meta/train_labeled.txt'
data_train_root = 'data/imagenet_raw/train/'
# data_train_list = 'data/cub/CUB_200_2011/meta/fine_cls/all.txt'
# data_train_root = 'data/cub/CUB_200_2011/images'

dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
data = dict(
    imgs_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline]))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='step', step=[120, 160])
# lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
