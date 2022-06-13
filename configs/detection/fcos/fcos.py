# model settings
model = dict(
    type='FCOS',
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/detectron/resnet50_caffe.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=[256, 256, 256, 256, 256],
        fpn_strides=[8, 16, 32, 64, 128],
        num_cls_convs=4,
        num_bbox_convs=4,
        num_share_convs=0,
        use_scale=True,
        fcos_outputs_config={
            'loss_alpha': 0.25,
            'loss_gamma': 2.0,
            'center_sample': True,
            'radius': 1.5,
            'pre_nms_thresh_train': 0.05,
            'pre_nms_topk_train': 1000,
            'post_nms_topk_train': 100,
            'loc_loss_type': 'giou',
            'pre_nms_thresh_test': 0.05,
            'pre_nms_topk_test': 1000,
            'post_nms_topk_test': 100,
            'nms_thresh': 0.6,
            'thresh_with_ctr': False,
            'box_quality': 'ctrness',
            'num_classes': 80,
            'strides': [8, 16, 32, 64, 128],
            'sizes_of_interest': [64, 128, 256, 512],
            'loss_normalizer_cls': 'fg',
            'loss_weight_cls': 1.0
        }))
