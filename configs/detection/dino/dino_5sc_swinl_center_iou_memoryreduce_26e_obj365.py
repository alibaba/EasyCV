_base_ = [
    './dino_5sc_swinl.py',
    '../common/dataset/autoaug_obj365_val5k_detection.py',
    './dino_schedule_1x.py'
]

data = dict(
    imgs_per_gpu=2
)  # total 64 = 2(update_interval) * 2(node_num) * 8(gpu_num) * 2(batch_size)

# model settings
model = dict(
    head=dict(
        num_classes=365,
        dn_components=dict(dn_number=1000, dn_labelbook_size=365),
        use_centerness=True,
        use_iouaware=True,
        losses_list=['labels', 'boxes', 'centerness', 'iouaware'],
        transformer=dict(multi_encoder_memory=True),
        weight_dict=dict(loss_ce=2, loss_center=2, loss_iouaware=2)))

# optimizer
optimizer_config = dict(update_interval=1)

# learning policy
lr_config = dict(policy='step', step=[24])

total_epochs = 26

checkpoint_config = dict(interval=1)
