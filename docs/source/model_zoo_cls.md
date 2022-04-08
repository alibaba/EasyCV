# Classification Model Zoo

## Benchmarks

| Algorithm | Config                                           | Top-1 (%) | Top-5 (%) | Download                                                     |
| --------- | ------------------------------------------------------------ | --------- | --------- | ------------------------------------------------------------ |
| resnet50(raw) | [resnet50(raw)](../../configs/classification/cls/resnet/imagenet_resnet50_jpg.py) | 76.454    | 93.084    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnet/resnet50/epoch_100.pth) |
| resnet50(tfrecord) | [resnet50(tfrecord)](../../configs/classification/imagenet/imagenet_rn50_tfrecord.py) | 76.266    | 92.972    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnet/resnet50/epoch_100.pth) |
| resnet101 | [resnet101](../../configs/classification/cls/resnet/imagenet_resnet101_jpg.py) | 78.152    | 93.922    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnet/resnet101/epoch_100.pth) |
| resnet152 | [resnet152](../../configs/classification/cls/resnet/imagenet_resnet152_jpg.py) | 78.544    | 94.206    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnet/resnet152/epoch_100.pth) |
| resnext50-32x4d | [resnext50-32x4d](../../configs/classification/cls/resnext/imagenet_resnext50-32x4d_jpg.py) | 77.604    | 93.856    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnext/resnet50/epoch_100.pth) |
| resnext101-32x4d | [resnext101-32x4d](../../configs/classification/cls/resnext/imagenet_resnext101-32x4d_jpg.py) | 78.568    | 94.344    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnext/resnext50-32x4d/epoch_100.pth) |
| resnext101-32x8d | [resnext101-32x8d](../../configs/classification/cls/resnext/imagenet_resnext101-32x8d_jpg.py) | 79.468    | 94.434    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnext/resnext101-32x8d/epoch_100.pth) |
| resnext152-32x4d | [resnext152-32x4d](../../configs/classification/cls/resnext/imagenet_resnext152-32x4d_jpg.py) | 78.994    | 94.462    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnext/resnext152-32x4d/epoch_100.pth) |
| hrnetw18 | [hrnetw18](../../configs/classification/cls/hrnet/imagenet_hrnetw18_jpg.py) | 76.258    | 92.976    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw18/epoch_100.pth) |
| hrnetw30 | [hrnetw30](../../configs/classification/cls/hrnet/imagenet_hrnetw30_jpg.py) | 77.66    | 93.862    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw30/epoch_100.pth) |
| hrnetw32 | [hrnetw32](../../configs/classification/cls/hrnet/imagenet_hrnetw32_jpg.py) | 77.994    | 93.976    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw32/epoch_100.pth) |
| hrnetw40 | [hrnetw40](../../configs/classification/cls/hrnet/imagenet_hrnetw40_jpg.py) | 78.142    | 93.956    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw40/epoch_100.pth) |
| hrnetw44 | [hrnetw44](../../configs/classification/cls/hrnet/imagenet_hrnetw44_jpg.py) | 79.266    | 94.476    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw44/epoch_100.pth) |
| hrnetw48 | [hrnetw48](../../configs/classification/cls/hrnet/imagenet_hrnetw48_jpg.py) | 79.636    | 94.802    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/hrnet/hrnetw48/epoch_100.pth) |
| hrnetw64 | [hrnetw64](../../configs/classification/cls/hrnet/imagenet_hrnetw64_jpg.py) | 79.884    | 95.04    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/resnet/hrnetw64/epoch_100.pth) |
| vit-base-patch16 | [vit-base-patch16](../../configs/classification/cls/vit/imagenet_vit_base_patch16_224_jpg.py) | 76.082    | 92.026    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/vit/vit-base-patch16/epoch_300.pth) |
| swin-tiny-patch4-window7 | [swin-tiny-patch4-window7](../../configs/classification/cls/swint/imagenet_swin_tiny_patch4_window7_224_jpg.py) | 80.528    | 94.822    | [model](http://easyvision-test.oss-cn-shanghai.aliyuncs.com/yunji.cjy/pretrain/swint/swin-tiny-patch4-window7/epoch_300.pth) |
