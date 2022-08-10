_base_ = 'configs/base.py'

total_epochs = 100
work_dir = './work_dir_dino'
# oss io config

oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',
    buckets=['your_bucket'])

# model settings
model_output_dim = 65536

model = dict(
    type='DINO',
    train_preprocess=[
        'randomGrayScale', 'gaussianBlur', 'solarize'
    ],  # 2+6 view, has different augment pipeline, dino is complex
    backbone=dict(
        type='PytorchImageModelWrapper',
        # model_name='pit_xs_distilled_224',

        # swin(dynamic)
        # model_name = 'dynamic_swin_tiny_p4_w7_224',
        # model_name = 'dynamic_swin_small_p4_w7_224',
        # model_name = 'dynamic_swin_base_p4_w7_224',

        # deit(224)
        model_name='dynamic_deit_small_p16',

        # xcit(224)
        # model_name='xcit_small_12_p16',
        # model_name='xcit_medium_24_p16',
        # model_name='xcit_large_24_p8',

        # resnet
        # model_name = 'resnet50',
        # model_name = 'resnet18',
        # model_name = 'resnet34',
        # model_name = 'resnet101',
    ),

    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(type='DINONeck', in_dim=384, out_dim=model_output_dim),
    config=dict(
        # dino head setting
        # momentum_teacher = 0.9995, #0.9995 for batchsize=256
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,
        use_tfrecord_input=False,

        # dino loss settding
        out_dim=model_output_dim,
        local_crops_number=8,
        warmup_teacher_temp=0.04,  # temperature for sharp softmax
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        epochs=total_epochs,
    ))

data_source_cfg = dict(type='SSLSourceImageList')

data_train_list = 'data/imagenet_raw/meta/train.txt'
data_train_root = 'data/imagenet_raw/'
dataset_type = 'MultiViewDataset'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = []

data = dict(
    imgs_per_gpu=60,  # total 32*8=256
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        num_views=[1],
        pipelines=[train_pipeline]))

custom_hooks = [
    dict(
        type='DINOHook',
        momentum_teacher=0.996,  # 0.9995 for bs=256
        weight_decay=0.04,
        weight_decay_end=0.4,
    )
]

# optimizer
optimizer = dict(type='AdamW', lr=9.375e-4)

optimizer_config = dict(
    grad_clip=dict(max_norm=3),
    ignore_key=['last_layer'],
    ignore_key_epoch=[1],  # in dino, freeze_last_layer in first 1 epoch
    # update_interval=2
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=1)

# export config
export = dict(export_neck=False)
# checkpoint_sync_export=True
