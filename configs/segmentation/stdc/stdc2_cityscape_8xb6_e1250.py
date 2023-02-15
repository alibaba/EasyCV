_base_ = ['configs/segmentation/stdc/stdc1.py']

model = dict(
    backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/pretrain/stdc2_easycv.pth'
)
