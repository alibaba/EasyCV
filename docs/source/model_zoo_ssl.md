# Self-supervised Learning Model Zoo
## Pretrained models

### MAE

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [mae_vit_base_patch16_8xb64_400e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_base_patch16_8xb64_400e.py) | 400    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-400/pretrain_400.pth) |
| [mae_vit_base_patch16_8xb64_1600e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_base_patch16_8xb64_1600e.py) | 1600   | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-1600/pretrain_1600.pth) |
| [mae_vit_large_patch16_8xb32_1600e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_large_patch16_8xb32_1600e.py) | 1600   | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-l-1600/pretrain_1600.pth) |

### DINO

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [dino_deit_small_p16_8xb32_100e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/dino/dino_deit_small_p16_8xb32_100e_tfrecord.py) | 100    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/dino_deit_small/epoch_100.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/dino_deit_small/log.txt) |

### MoBY

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [moby_deit_small_p16_4xb128_300e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/moby/moby_deit_small_p16_4xb128_300e_tfrecord.py) | 300    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/moby_deit_small_p16/epoch_300.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/moby_deit_small_p16/log.txt) |

### MoCo V2

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [mocov2_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 200    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mocov2_r50/epoch_200.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mocov2_r50/log.txt) |

### SwAV

Pretrained on **ImageNet** dataset.


| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [swav_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 200    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/swav_r50/epoch_200.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/swav_r50/log.txt) |

## Benchmarks

For detailed usage of benchmark tools, please refer to benchmark [README.md](../../benchmarks/selfsup/README.md).

### ImageNet Linear Evaluation

| Algorithm | Linear Eval Config                                           | Pretrained Config                                            | Top-1 (%) | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
| SwAV      | [swav_resnet50_8xb2048_20e_feature](../../benchmarks/selfsup/classification/imagenet/swav_r50_8xb2048_20e_feature.py) | [swav_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 73.618    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/swav_r50_linear_eval/20220216_101719.log.json) |
| DINO      | [dino_deit_small_p16_8xb2048_20e_feature](../../benchmarks/selfsup/classification/imagenet/dino_deit_small_p16_8xb2048_20e_feature.py) | [dino_deit_small_p16_8xb32_100e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/dino/dino_deit_small_p16_8xb32_100e_tfrecord.py) | 71.248    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/dino_deit_small_linear_eval/20220215_141403.log.json) |
| MoBY | [moby_deit_small_p16_8xb2048_30e_feature](../../benchmarks/selfsup/classification/imagenet/moby_deit_small_p16_8xb2048_30e_feature.py) | [moby_deit_small_p16_4xb128_300e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/moby/moby_deit_small_p16_4xb128_300e_tfrecord.py) | 72.214    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/moby_deit_small_p16_linear_eval/20220414_134929.log.json) |
| MoCo-v2   | [mocov2_resnet50_8xb2048_40e_feature](../../benchmarks/selfsup/classification/imagenet/mocov2_r50_8xb2048_40e_feature.py) | [mocov2_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 66.8      | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/mocov2_r50_linear_eval/20220214_143738.log.json) |

### ImageNet Finetuning

| Algorithm | Fintune Config                                               | Pretrained Config                                            | Top-1 (%) | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
| **MAE**   | [mae_vit_base_patch16_8xb64_100e_lrdecay075_fintune](../../benchmarks/selfsup/classification/imagenet/mae_vit_base_patch16_8xb64_100e_lrdecay075_fintune.py) | [mae_vit_base_patch16_8xb64_400e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_base_patch16_8xb64_400e.py) | 83.13     | [fintune model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-400/fintune_400.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-400/20220126_171312.log.json)|
|           | [mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune](../../benchmarks/selfsup/classification/imagenet/mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune.py) | [mae_vit_base_patch16_8xb64_1600e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_base_patch16_8xb64_1600e.py) | 83.55     | [fintune model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-1600/fintune_1600.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-1600/20220426_101532.log.json)|
|           | [mae_vit_large_patch16_8xb16_50e_lrdecay075_fintune](../../benchmarks/selfsup/classification/imagenet/mae_vit_large_patch16_8xb16_50e_lrdecay075_fintune.py) | [mae_vit_large_patch16_8xb32_1600e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mae/mae_vit_large_patch16_8xb32_1600e.py) | 85.70     | [fintune model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-l-1600/fintune_1600.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-l-1600/20220427_150629.log.json)|

### COCO2017 Object Detection

| Algorithm | Eval Config                                                  | Pretrained Config                                            | mAP (Box) | mAP (Mask) | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ---------- | ------------------------------------------------------------ |
| SwAV      | [mask_rcnn_r50_fpn_1x_coco](https://github.com/alibaba/EasyCV/tree/master/benchmarks/selfsup/detection/coco/mask_rcnn_r50_fpn_1x_coco.py) | [swav_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 40.38     | 36.48      | [eval model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/detection/mask_rcnn_r50_fpn/mocov2_r50/epoch_12.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/detection/mask_rcnn_r50_fpn/mocov2_r50/20220510_164934.log.json) |
| MoCo-v2   | [mask_rcnn_r50_fpn_1x_coco](https://github.com/alibaba/EasyCV/tree/master/benchmarks/selfsup/detection/coco/mask_rcnn_r50_fpn_1x_coco.py) | [mocov2_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 39.9     | 35.8      | [eval model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/detection/mask_rcnn_r50_fpn/swav_r50/epoch_12.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/detection/mask_rcnn_r50_fpn/swav_r50/20220513_142102.log.json) |

### VOC2012 Aug Semantic Segmentation

| Algorithm | Eval Config                                                  | Pretrained Config                                            | mIOU  | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| SwAV      | [fcn_r50-d8_512x512_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/benchmarks/selfsup/segmentation/voc/fcn_r50-d8_512x512_8xb4_60e_voc12aug.py) | [swav_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 63.91 | [eval model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/segmentation/swav_fcn_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/segmentation/swav_fcn_r50/20220525_171032.log.json) |
| MoCo-v2   | [fcn_r50-d8_512x512_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/benchmarks/selfsup/segmentation/voc/fcn_r50-d8_512x512_8xb4_60e_voc12aug.py) | [mocov2_resnet50_8xb32_200e](https://github.com/alibaba/EasyCV/tree/master/configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 68.49 | [eval model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/segmentation/mocov2_fcn_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/benchmarks/segmentation/mocov2_fcn_r50/20220525_211410.log.json) |
