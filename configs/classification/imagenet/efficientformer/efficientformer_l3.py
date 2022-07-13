_base_ = './efficientformer_l1.py'
# model settings
model = dict(
    pretrained='/home/yunji.cjy/pretrain/efficientformer_l3_300d.pth',
    backbone=dict(
        layers=[4, 4, 12, 6], embed_dims=[64, 128, 320, 512], vit_num=4))
