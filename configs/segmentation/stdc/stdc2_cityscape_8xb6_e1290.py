_base_ = ['configs/segmentation/stdc/stdc1_cityscape_8xb6_e1290.py']

model = dict(
    backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/pretrain/stdc2_easycv.pth'
)
