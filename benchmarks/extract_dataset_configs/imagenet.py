_base_ = 'configs/base.py'

data_all_list = 'data/imagenet_raw/meta/all_labeled.txt'
data_root = 'data/imagenet_raw/'

total_samples_num = 1281167
part_num = 2048
split_at = [*range(0, total_samples_num, part_num), total_samples_num]
split_name = [*['train_idx%d' % i for i in range(len(split_at))], 'val']

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=4,
    extract=dict(
        type='RawDataset',
        data_source=dict(
            type='ClsSourceImageList', list_file=data_all_list,
            root=data_root),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Collect', keys=['img', 'gt_labels'])
        ]))
