_base_ = 'configs/base.py'

data_all_list = 'data/cub/CUB_200_2011/meta/fine_cls/all.txt'
data_root = 'data/cub/CUB_200_2011/images/'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=2,
    extract=dict(
        type='RawDataset',
        with_label=True,
        data_source=dict(
            type='ClsSourceImageList', list_file=data_all_list,
            root=data_root),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))

# extract info
# split_name = ["cub_train", "cub_val"]
total_samples_num = 8565
part_num = 100
split_at = [*range(0, total_samples_num, part_num), total_samples_num]
split_name = [*['train_idx%d' % i for i in range(len(split_at))], 'val']
