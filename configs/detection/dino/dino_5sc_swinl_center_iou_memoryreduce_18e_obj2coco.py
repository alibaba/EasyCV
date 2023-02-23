_base_ = [
    './dino_5sc_swinl.py', '../common/dataset/autoaug_obj2coco_detection.py',
    './dino_schedule_1x.py'
]

data = dict(imgs_per_gpu=1)  # total 16 = 8(gpu_num) * 2(batch_size)

# model settings
model = dict(
    head=dict(
        dn_components=dict(dn_number=1000),
        use_centerness=True,
        use_iouaware=True,
        losses_list=['labels', 'boxes', 'centerness', 'iouaware'],
        transformer=dict(multi_encoder_memory=True),
        weight_dict=dict(loss_ce=2, loss_center=2, loss_iouaware=2)))

# learning policy
lr_config = dict(policy='step', step=[8])

total_epochs = 18

checkpoint_config = dict(interval=1)

load_form = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_5sc_swinl_obj365/epoch_22.pth'
