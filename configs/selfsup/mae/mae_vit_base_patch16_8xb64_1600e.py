_base_ = '../../base.py'

# open oss config when using oss
# sync local models and logs to oss
# oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])
# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='your oss hosts',
#     buckets=['your oss buckets'])

# model setting
model = dict(
    type='MAE',
    backbone=dict(
        type='MaskedAutoencoderViT',
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
    ),
    neck=dict(
        type='MAENeck',
        embed_dim=768,
        patch_size=16,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    mask_ratio=0.75,
    norm_pix_loss=True)

# dataset settings
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet'

dataset_type = 'RawDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.0),
        interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='SSLSourceImageList',
            list_file=data_train_list,
            root=data_train_root),
        pipeline=train_pipeline))

# optimizer
update_interval = 8
optimizer_config = dict(update_interval=update_interval)
eff_batch_size = 64 * 8 * update_interval  # 4096
optimizer = dict(
    type='AdamW',
    lr=1.5e-4 * eff_batch_size / 256,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=40,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)
checkpoint_config = dict(interval=50)

# runtime settings
total_epochs = 1600
