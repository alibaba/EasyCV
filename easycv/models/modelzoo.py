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
    'vit_base_patch16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/vit/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    'vit_large_patch16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/vit/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    'deit_base_patch16_224':
    'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
    'deit_base_distilled_patch16_224':
    'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    'swin_base_patch4_window7_224':
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    'swin_large_patch4_window7_224':
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    'xcit_medium_24_p8_224':
    'https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224.pth',
    'xcit_medium_24_p8_224_dist':
    'https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p8_224_dist.pth',
    'xcit_large_24_p8_224':
    'https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224.pth',
    'xcit_large_24_p8_224_dist':
    'https://dl.fbaipublicfiles.com/xcit/xcit_large_24_p8_224_dist.pth',
    'twins_svt_small':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_small-42e5f78c.pth',
    'twins_svt_base':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_base-c2265010.pth',
    'twins_svt_large':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_large-90f6aaa9.pth',
    'tnt_s_patch16_224':
    'https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
    'pit_s_distilled_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_s_distill_819.pth',
    'pit_b_distilled_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_b_distill_840.pth',
    'jx_nest_tiny':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/jx_nest_tiny-e3428fb9.pth',
    'jx_nest_small':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/jx_nest_small-422eaded.pth',
    'jx_nest_base':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/jx_nest_base-8bc41011.pth',
    'crossvit_tiny_240':
    'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
    'crossvit_small_240':
    'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_base_240':
    'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'convit_tiny':
    'https://dl.fbaipublicfiles.com/convit/convit_tiny.pth',
    'convit_small':
    'https://dl.fbaipublicfiles.com/convit/convit_small.pth',
    'convit_base':
    'https://dl.fbaipublicfiles.com/convit/convit_base.pth',
    'coat_tiny':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_tiny-473c2a20.pth',
    'coat_mini':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_mini-2c6baf49.pth',
    'cait_xxs24_224':
    'https://dl.fbaipublicfiles.com/deit/XXS24_224.pth',
    'cait_xxs36_224':
    'https://dl.fbaipublicfiles.com/deit/XXS36_224.pth',
    'cait_s24_224':
    'https://dl.fbaipublicfiles.com/deit/S24_224.pth',
    'levit_128':
    'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth',
    'levit_192':
    'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth',
    'levit_256':
    'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth',
    'convmixer_1536_20':
    'https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1536_20_ks9_p7.pth.tar',
    'convmixer_768_32':
    'https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_768_32_ks7_p7_relu.pth.tar',
    'convmixer_1024_20_ks9_p14':
    'https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1024_20_ks9_p14.pth.tar',
    'convnext_tiny':
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
    'convnext_small':
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
    'convnext_base':
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth',
    'convnext_large':
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth',
    'mixer_b16_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    'mixer_l16_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    'gmixer_24_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth',
    'resmlp_12_distilled_224':
    'https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
    'resmlp_24_distilled_224':
    'https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
    'resmlp_36_distilled_224':
    'https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
    'resmlp_big_24_distilled_224':
    'https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
    'gmlp_s16_224':
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
}
