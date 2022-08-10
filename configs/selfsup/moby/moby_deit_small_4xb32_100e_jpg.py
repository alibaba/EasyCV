_base_ = 'configs/base.py'  # _base_ = 'configs/base.py'

# model settings
model = dict(
    type='MoBY',
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    queue_len=4096,
    momentum=0.99,
    pretrained=False,
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',
        # model_name='swin_small_patch4_window7_224',         # bad 16G memory will down
        # model_name='swin_tiny_patch4_window7_224',          # good 768
        # model_name='swin_base_patch4_window7_224_in22k',    # bad 16G memory will down
        # model_name='vit_deit_tiny_distilled_patch16_224',   # good 192,
        model_name='vit_deit_small_distilled_patch16_224',  # good 384,
        # model_name = 'resnet50',
    ),
    neck=dict(
        type='MoBYMLP',
        in_channels=384,
        hid_channels=4096,
        out_channels=256,
        num_layers=2),
    head=dict(
        type='MoBYMLP',
        in_channels=256,
        hid_channels=4096,
        out_channels=256,
        num_layers=2))

data_train_list = 'imagenet_raw/meta/train.txt'
data_train_root = 'imagenet_raw/'

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
        type='MultiViewDataset',
        data_source=dict(
            type='SSLSourceImageList',
            list_file=data_train_list,
            root=data_train_root),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline]))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    trans_weight_decay_set=['backbone'])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=5e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=5)

# runtime settings
total_epochs = 100

# export config
# export = dict(export_neck=True)
export = dict(export_neck=False)
checkpoint_sync_export = True
