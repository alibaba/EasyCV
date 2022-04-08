# Self-supervised Learning Model Zoo
## Pretrained models

### MAE

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [mae_vit_base_patch16_8xb64_400e](../../configs/selfsup/mae/mae_vit_base_patch16_8xb64_400e.py) | 400    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-400/pretrain_400.pth) |
| [mae_vit_base_patch16_8xb64_1600e](../../configs/selfsup/mae/mae_vit_base_patch16_8xb64_1600e.py) | 1600   | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-1600/pretrain_1600.pth) |

### DINO

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [dino_deit_small_p16_8xb32_100e](../../configs/selfsup/dino/dino_deit_small_p16_8xb32_100e_tfrecord.py) | 100    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/dino_deit_small/epoch_100.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/dino_deit_small/log.txt) |

### MoBY

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [moby_resnet50_4xb128_100e](../../configs/selfsup/moby/moby_rn50_4xb128_100e_tfrecord.py) | 100    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/moby_r50/epoch_100.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/moby_r50/log.txt) |

### MoCo V2

Pretrained on **ImageNet** dataset.

| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [mocov2_resnet50_8xb32_200e](../../configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 200    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mocov2_r50/epoch_200.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mocov2_r50/log.txt) |

### SwAV

Pretrained on **ImageNet** dataset.


| Config                                                       | Epochs | Download                                                     |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [swav_resnet50_8xb32_200e](../../configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 200    | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/swav_r50/epoch_200.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/swav_r50/log.txt) |

## Benchmarks

For detailed usage of benchmark tools, please refer to benchmark [README.md](../../benchmarks/selfsup/README.md).

### ImageNet Linear Evaluation

| Algorithm | Linear Eval Config                                           | Pretrained Config                                            | Top-1 (%) | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
| SwAV      | [swav_resnet50_8xb2048_20e_feature](../../benchmarks/selfsup/classification/imagenet/swav_r50_8xb2048_20e_feature.py) | [swav_resnet50_8xb32_200e](../../configs/selfsup/swav/swav_rn50_8xb32_200e_tfrecord.py) | 73.618    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/swav_r50_linear_eval/20220216_101719.log.json) |
| DINO      | [dino_deit_small_p16_8xb2048_20e_feature](../../benchmarks/selfsup/classification/imagenet/dino_deit_small_p16_8xb2048_20e_feature.py) | [dino_deit_small_p16_8xb32_100e](../../configs/selfsup/dino/dino_deit_small_p16_8xb32_100e_tfrecord.py) | 71.248    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/dino_deit_small_linear_eval/20220215_141403.log.json) |
| MoBY r50  | [moby_resnet50_8xb2048_20e_feature](../../benchmarks/selfsup/classification/imagenet/moby_r50_8xb2048_20e_feature.py) | [moby_resnet50_4xb128_100e](../../configs/selfsup/moby/moby_rn50_4xb128_100e_tfrecord.py) | 78.392    | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/moby_r50_linear_eval/20220214_135816.log.json) |
| MoCo-v2   | [mocov2_resnet50_8xb2048_40e_feature](../../benchmarks/selfsup/classification/imagenet/mocov2_r50_8xb2048_40e_feature.py) | [mocov2_resnet50_8xb32_200e](../../configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py) | 66.8      | [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/imagenet_linear_eval/mocov2_r50_linear_eval/20220214_143738.log.json) |

### ImageNet Finetuning

| Algorithm | Fintune Config                                               | Pretrained Config                                            | Top-1 (%) | Download                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
| **MAE**   | [mae_vit_base_patch16_8xb64_100e_lrdecay075_fintune](../../benchmarks/selfsup/classification/imagenet/mae_vit_base_patch16_8xb64_100e_lrdecay075_fintune.py) | [mae_vit_base_patch16_8xb64_400e](../../configs/selfsup/mae/mae_vit_base_patch16_8xb64_400e.py) | 83.13     | [fintune model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-400/fintune_400.pth) |
|           | [mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune](../../benchmarks/selfsup/classification/imagenet/mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune.py) | [mae_vit_base_patch16_8xb64_1600e](../../configs/selfsup/mae/mae_vit_base_patch16_8xb64_1600e.py) | 83.49     | [fintune model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/selfsup/mae/vit-b-1600/fintune_1600.pth) |
