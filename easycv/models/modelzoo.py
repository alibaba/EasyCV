# Copyright (c) Alibaba, Inc. and its affiliates.
resnet = {
    'ResNet18':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet18.pth',
    'ResNet34':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet34.pth',
    'ResNet50':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet50.pth',
    'ResNet101':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet101.pth',
    'ResNet152':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet152.pth',
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

mobilenetv2 = {
    'MobileNetV2_1.0':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/mobilenetv2/mobilenet_v2.pth',
}

mnasnet = {
    'MNASNet0.5':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/mnasnet/mnasnet0.5.pth',
    'MNASNet0.75': None,
    'MNASNet1.0':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/mnasnet/mnasnet1.0.pth',
    'MNASNet1.3': None
}

inceptionv3 = {
    # Inception v3 ported from TensorFlow
    'Inception3':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/inceptionv3/inception_v3.pth',
}

genet = {
    'PlainNetnormal':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/genet/GENet_normal.pth',
    'PlainNetlarge':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/genet/GENet_large.pth',
    'PlainNetsmall':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/genet/GENet_small.pth'
}

bninception = {
    'BNInception':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/bninception/bn_inception.pth',
}

resnest = {
    'ResNeSt50':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnest/resnest50.pth',
    'ResNeSt101':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnest/resnest101.pth',
    'ResNeSt200':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnest/resnest200.pth',
    'ResNeSt269':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnest/resnest269.pth',
}

timm_models = {
    'resnet50':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/resnet50_ram-a26f946b.pth',
    'resnet18':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet18.pth',
    'resnet34':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet34.pth',
    'resnet101':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet101.pth',
    'resnet152':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet152.pth',
    'swin_base_patch4_window7_224_in22k':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_base_patch4_window7_224_22k_statedict.pth',
    'swin_small_patch4_window7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_small_patch4_window7_224_statedict.pth',
    'swin_tiny_patch4_window7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_tiny_patch4_window7_224_statedict.pth',
    'vit_deit_tiny_distilled_patch16_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/vit_deit_tiny_distilled_patch16_224_pytorch151.pth',
    'vit_deit_small_distilled_patch16_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/vit_deit_small_distilled_patch16_224_pytorch151.pth',

    # facebook xcit
    'xcit_small_12_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_small_12_p16_pretrain.pth',  # 384
    'xcit_small_12_p8':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_small_12_p8_pretrain.pth',  # 384
    'xcit_medium_24_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_medium_24_p16_pretrain.pth',  # 512
    'xcit_medium_24_p8':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_medium_24_p8_pretrain.pth',  # 512
    'xcit_large_24_p8':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/xcit_large_24_p8_224.pth',

    # shuffle_trans
    'shuffletrans_base_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/imagenet/shuffle_transformer/shuffle_base.pth',
    'shuffletrans_small_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/imagenet/shuffle_transformer/shuffle_small.pth',
    'shuffletrans_tiny_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/imagenet/shuffle_transformer/shuffle_tiny.pth',

    # dynamic swint:
    'dynamic_swin_base_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_base_patch4_window7_224_22k_statedict.pth',
    'dynamic_swin_small_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_small_patch4_window7_224_statedict.pth',
    'dynamic_swin_tiny_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_tiny_patch4_window7_224_statedict.pth',

    # dynamic vit:
    'dynamic_deit_tiny_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/vit_deit_tiny_distilled_patch16_224_pytorch151.pth',
    'dynamic_deit_small_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/vit_deit_small_distilled_patch16_224_pytorch151.pth',
    'resnet50':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/resnet50_ram-a26f946b.pth',
    'resnet18':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet18.pth',
    'resnet34':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet34.pth',
    'resnet101':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet101.pth',
    'resnet152':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet152.pth',
}
