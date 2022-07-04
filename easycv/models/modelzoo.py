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

resnext = {
    'ResNeXt50-32x4d':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/resnext/resnext50-32x4d/epoch_100.pth',
    'ResNeXt101-32x4d':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/resnext/resnext101-32x4d/epoch_100.pth',
    'ResNeXt101-32x8d':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/resnext/resnext101-32x8d/epoch_100.pth',
    'ResNext152-32x4d':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/resnext/resnext152-32x4d/epoch_100.pth',
}

hrnet = {
    'HRNetw18':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw18/epoch_100.pth',
    'HRNetw30':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw30/epoch_100.pth',
    'HRNetw32':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw32/epoch_100.pth',
    'HRNetw40':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw40/epoch_100.pth',
    'HRNetw44':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw44/epoch_100.pth',
    'HRNetw48':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw48/epoch_100.pth',
    'HRNetw64':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hrnet/hrnetw64/epoch_100.pth',
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
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/deit/deit_base_patch16_224-b5f2ef4d.pth',
    'deit_base_distilled_patch16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    'swin_tiny_patch4_window7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_tiny_patch4_window7_224.pth',
    'swin_small_patch4_window7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_small_patch4_window7_224.pth',
    'swin_base_patch4_window7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_base_patch4_window7_224_22kto1k.pth',
    'swin_large_patch4_window7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_large_patch4_window7_224_22kto1k.pth',
    'swin_v2_cr_small_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_v2_cr_small_224-0813c165.pth',
    'swin_v2_cr_small_ns_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_v2_cr_small_ns_224_iv-2ce90f8e.pth',
    'swin_v2_cr_tiny_ns_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/swint/swin_v2_cr_tiny_ns_224-ba8166c6.pth',
    'xcit_medium_24_p8_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/xcit/xcit_medium_24_p8_224.pth',
    'xcit_medium_24_p8_224_dist':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/xcit/xcit_medium_24_p8_224_dist.pth',
    'xcit_large_24_p8_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/xcit/xcit_large_24_p8_224.pth',
    'xcit_large_24_p8_224_dist':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/xcit/xcit_large_24_p8_224_dist.pth',
    'twins_svt_small':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/twins/twins_svt_small-42e5f78c.pth',
    'twins_svt_base':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/twins/twins_svt_base-c2265010.pth',
    'twins_svt_large':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/twins/twins_svt_large-90f6aaa9.pth',
    'tnt_s_patch16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/tnt/tnt_s_patch16_224.pth.tar',
    'pit_s_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/pit/pit_s_distill_819.pth',
    'pit_b_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/pit/pit_b_distill_840.pth',
    'jx_nest_tiny':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/nest/jx_nest_tiny-e3428fb9.pth',
    'jx_nest_small':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/nest/jx_nest_small-422eaded.pth',
    'jx_nest_base':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/nest/jx_nest_base-8bc41011.pth',
    'crossvit_tiny_240':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/crossvit/crossvit_tiny_224.pth',
    'crossvit_small_240':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/crossvit/crossvit_small_224.pth',
    'crossvit_base_240':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/crossvit/crossvit_base_224.pth',
    'convit_tiny':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convit/convit_tiny.pth',
    'convit_small':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convit/convit_small.pth',
    'convit_base':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convit/convit_base.pth',
    'coat_tiny':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/coat/coat_tiny-473c2a20.pth',
    'coat_mini':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/coat/coat_mini-2c6baf49.pth',
    'cait_xxs24_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/cait/XXS24_224.pth',
    'cait_xxs36_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/cait/XXS36_224.pth',
    'cait_s24_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/cait/S24_224.pth',
    'levit_128':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/levit/LeViT-128-b88c2750.pth',
    'levit_192':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/levit/LeViT-192-92712e41.pth',
    'levit_256':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/levit/LeViT-256-13b5763e.pth',
    'convmixer_1536_20':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convmixer/convmixer_1536_20_ks9_p7.pth.tar',
    'convmixer_768_32':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convmixer/convmixer_768_32_ks7_p7_relu.pth.tar',
    'convmixer_1024_20_ks9_p14':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convmixer/convmixer_1024_20_ks9_p14.pth.tar',
    'convnext_tiny':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convnext/convnext_tiny_1k_224_ema.pth',
    'convnext_small':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convnext/convnext_small_1k_224_ema.pth',
    'convnext_base':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convnext/convnext_base_1k_224_ema.pth',
    'convnext_large':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/convnext/convnext_large_1k_224_ema.pth',
    'mixer_b16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/mlp-mixer/jx_mixer_b16_224-76587d61.pth',
    'mixer_l16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/mlp-mixer/jx_mixer_l16_224-92f9adc4.pth',
    'gmixer_24_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/gmixer/gmixer_24_224_raa-7daf7ae6.pth',
    'resmlp_12_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/resmlp/resmlp_12_dist.pth',
    'resmlp_24_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/resmlp/resmlp_24_dist.pth',
    'resmlp_36_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/resmlp/resmlp_36_dist.pth',
    'resmlp_big_24_distilled_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/resmlp/resmlpB_24_dist.pth',
    'gmlp_s16_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/gmlp/gmlp_s16_224_raa-10536d42.pth',
    'mobilevit_xxs':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/mobilevit/mobilevit_xxs-ad385b40.pth',
    'mobilevit_xs':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/mobilevit/mobilevit_xs-8fbd6366.pth',
    'mobilevit_s':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/mobilevit/mobilevit_s-38a5a959.pth',
    'poolformer_s12':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/poolformer/poolformer_s12.pth.tar',
    'poolformer_s24':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/poolformer/poolformer_s24.pth.tar',
    'poolformer_s36':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/poolformer/poolformer_s36.pth.tar',
    'poolformer_m36':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/poolformer/poolformer_m36.pth.tar',
    'poolformer_m48':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/poolformer/poolformer_m48.pth.tar',
    'volo_d1_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/volo/d1_224_84.2.pth.tar',
    'volo_d2_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/volo/d2_224_85.2.pth.tar',
    'volo_d3_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/volo/d3_224_85.4.pth.tar',
    'volo_d4_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/volo/d4_224_85.7.pth.tar',
    'volo_d5_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/timm/volo/d5_224_86.10.pth.tar',

    # facebook xcit
    'xcit_small_12_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_small_12_p16_pretrain.pth',  # 384
    'xcit_small_12_p8':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_small_12_p8_pretrain.pth',  # 384
    'xcit_medium_24_p16':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_medium_24_p16_pretrain.pth',  # 512
    'xcit_medium_24_p8':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/xcit/dino_xcit_medium_24_p8_pretrain.pth',  # 512

    # shuffle_trans
    'shuffletrans_base_p4_w7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/shuffle_transformer/shuffle_base.pth',
    'shuffletrans_small_p4_w7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/shuffle_transformer/shuffle_small.pth',
    'shuffletrans_tiny_p4_w7_224':
    'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/shuffle_transformer/shuffle_tiny.pth',

    # dynamic swint:
    'dynamic_swin_base_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_base_patch4_window7_224_22k_statedict.pth',
    'dynamic_swin_small_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_small_patch4_window7_224_statedict.pth',
    'dynamic_swin_tiny_p4_w7_224':
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/timm/swin_tiny_patch4_window7_224_statedict.pth',
}
