_base_ = 'configs/base.py'
# model settings

model = dict(
    type='MOCO',
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',
        # model_name='swin_small_patch4_window7_224',         # bad 16G memory will down
        # model_name='swin_tiny_patch4_window7_224',          # good 768
        # model_name='swin_base_patch4_window7_224_in22k',    # bad 16G memory will down
        # model_name='vit_deit_tiny_distilled_patch16_224',   # good 192,
        # model_name = 'vit_deit_small_distilled_patch16_224', # good 384,
        # model_name = 'resnet50',
        # model_name='xcit_small_12_p8',  #384
        # model_name='xcit_medium_24_p8',  #384
        model_name='xcit_large_24_p8',  # 768
        num_classes=0,
    ),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=768,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = [x * 255 for x in img_norm_cfg['mean']]
std = [x * 255 for x in img_norm_cfg['std']]
train_pipeline = [
    dict(type='DaliImageDecoder'),
    dict(type='DaliRandomResizedCrop', size=224, random_area=(0.2, 1.0)),
    dict(
        type='DaliColorTwist',
        prob=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(
        type='DaliCropMirrorNormalize',
        crop=[224, 224],
        mean=mean,
        std=std,
        crop_pos_x=[0.0, 1.0],
        crop_pos_y=[0.0, 1.0],
        prob=0.5)
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=2,
    train=dict(
        type='DaliTFRecordMultiViewDataset',
        data_source=dict(
            type='ClsSourceImageNetTFRecord',
            file_pattern='oss://path/to/data/imagenet_tfrecord/train-*',
            cache_path='data/imagenet_tfrecord'),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline],
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
load_from = None
resume_from = None

# export config
export = dict(export_neck=False)
checkpoint_sync_export = True
