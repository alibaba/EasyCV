_base_ = 'configs/base.py'
# model settings
model = dict(
    type='Classification',
    train_preprocess=['randomErasing'],
    pretrained=
    'oss://pai-vision-data-hz/EasyCV/modelzoo/imagenet/shuffle_transformer/shuffle_tiny.pth',
    backbone=dict(type='ShuffleTransformer', qkv_bias=True, num_classes=0),
    # backbone=dict(
    #     type='PytorchImageModelWrapper',
    #     #model_name='pit_xs_distilled_224',
    #     #model_name='swin_small_patch4_window7_224',
    #     model_name='swin_tiny_patch4_window7_224',
    #     #model_name='swin_base_patch4_window7_224_in22k',
    #     #model_name='vit_deit_small_distilled_patch16_224',
    #     #model_name = 'vit_deit_small_distilled_patch16_224',
    #     #model_name = 'resnet50',
    #     #num_classes=768,
    #     pretrained=True,
    # ),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        with_fc=True,
        in_channels=768,
        num_classes=201,
        label_smooth=0.1))

data_train_list = 'data/cub/CUB_200_2011/meta/fine_cls/train.txt'
data_train_root = 'data/cub/CUB_200_2011/images'
data_test_list = 'data/cub/CUB_200_2011/meta/fine_cls/val.txt'
data_test_root = 'data/cub/CUB_200_2011/images'
dataset_type = 'ClsDataset'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            type='ClsSourceImageList'),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
val_data_list = dict(
    imgs_per_gpu=20,
    workers_per_gpu=4,
    val=dict(
        type='ClsDataset',
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type='ClsSourceImageList'),
        pipeline=test_pipeline))

eval_config = dict(interval=1, gpu_collect=True, initial=True)
eval_pipelines = [
    dict(
        mode='test',
        data=val_data_list['val'],
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# additional hooks
custom_hooks = []
# optimizer
# optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0.05, set_var_bias_nowd=['backbone']) # tran_weight_decay_set, for some transformer, some bias should not count in wd
optimizer = dict(
    type='AdamW', lr=0.0005, weight_decay=0.05
)  # set_var_bias_nowd=['backbone']) # tran_weight_decay_set, for some transformer, some bias should not count in wd
# optimizer = dict(type='SGD', lr=0.1, weight_decay=2e-5, set_var_bias_nowd=['backbone']) # tran_weight_decay_set, for some transformer, some bias should not count in wd
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0005,
    warmup='linear',
    warmup_iters=3,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)

checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 30
