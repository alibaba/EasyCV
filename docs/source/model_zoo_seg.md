# Segmentation Model Zoo

## FCN

Pretrained on **Pascal VOC 2012 + Aug**.

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | Train memory<br/>(GB)      | inference time(V100)<br/>(ms/img)                     | mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fcn_r50_d8 | [fcn_r50-d8_512x512_8xb4_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/fcn/fcn_r50-d8_512x512_8xb4_60e_voc12aug.py) | 23M/49M | 19.8 | 166ms | 69.01               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/fcn_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/fcn_r50/20220525_203606.log.json) |

## UperNet

Pretrained on **Pascal VOC 2012 + Aug**.
| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | Train memory<br/>(GB)       | inference time(V100)<br/>(ms/img)                      | mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| upernet_r50 | [upernet_r50_512x512_8xb4_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/upernet/upernet_r50_512x512_8xb4_60e_voc12aug.py) | 23M/66M | 5.5 | 282.9ms | 76.59               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/upernet_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/upernet_r50/20220706_114712.log.json) |

## STDC
trained on **Cityscapes**.
| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | Train memory<br/>(GB)       | inference time(V100)<br/>(ms/img)                      | mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| STDC1 | [stdc1_cityscape_8xb6_e1290](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/stdc/stdc1_cityscape_8xb6_e1290.py) | 7.7M/8.5M | 4.5 | 11.9ms | 75.4               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/stdc1_cityscapes/epoch_1250.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/stdc1_cityscapes/20230214_173123.log.json) |
| STDC2 | [stdc2_cityscape_8xb6_e1290](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/stdc/stdc2_cityscape_8xb6_e1290.py) | 11.6M/12.6M | 5.6 | 15.4ms | 76.6               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/stdc2_cityscapes/epoch_1280.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/stdc2_cityscapes/20230216_110522.log.json) |

## Mask2former

### Instance Segmentation on COCO
| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | Train memory<br/>(GB)                                  | inference time(A100)<br/>(ms/img)                     | box MAP | Mask mAP | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ |----------|----------|----------|----------|---|
| mask2former_r50 | [mask2former_r50_8xb2_e50_instance](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/mask2former/mask2former_r50_8xb2_e50_instance.py) | 23.5M/44M | 18.8 | 214ms | 46.09 | 43.26 |[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_instance/epoch_50.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_instance/20220620_113639.log.json) |

### Panoptic Segmentation on COCO

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | Train memory<br/>(GB)                                  | inference time(A100)<br/>(ms/img)                     | PQ | box MAP | Mask mAP | Download                                                     |
| ---------- | ---------- | ------------------------------------------------------------ | ------------------------ |----------|---------------------------------------------------------------------------- |---------------------------------------------------------------------------- |---|---|
| mask2former_r50 | [mask2former_r50_8xb2_e50_panopatic](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/mask2former/mask2former_r50_8xb2_e50_panopatic.py) | 23.5M/44M | 18.8 | 241ms | 51.64 | 44.81 | 41.88 |[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_panoptic/epoch_50.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_panoptic/20220629_170721.log.json) |

### Semantic Segmentation on ADE20K

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                             |Train memory<br/>(GB)                                  | inference time(A100)<br/>(ms/img)|                      mIOU |Download                                                     |
| ---------- | ---------- | ------------------------------------------------------------ |---------------------------------------------------------------------------- |---------------------------------------------------------------------------- |---|---|
| mask2former_r50 | [mask2former_r50_8xb2_e127_semantic](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/mask2former/mask2former_r50_8xb2_e127_semantic.py) | 23.5M/44M | 5.6 | 504ms | 47.03 |[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_semantic/epoch_116.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_semantic/20220929_145919.log.json) |

## SegFormer

Semantic segmentation models trained on **CoCo_stuff164k**.

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | inference time(V100)<br/>(ms/img)                    |mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SegFormer_B0 | [segformer_b0_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b0_coco.py) | 3.3M/3.8M | 47.2ms |  35.91               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b0/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b0/20220909_152337.log.json) |
| SegFormer_B1 | [segformer_b1_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b1_coco.py) | 13.2M/13.7M | 46.8ms |  40.53               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b1/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b1/20220825_200708.log.json) |
| SegFormer_B2 | [segformer_b2_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b2_coco.py) | 24.2M/27.5M   | 49.1ms |  44.53               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b2/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b2/20220829_163757.log.json) |
| SegFormer_B3 | [segformer_b3_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b3_coco.py) | 44.1M/47.4M | 52.3ms |  45.49               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b3/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b3/20220830_142021.log.json) |
| SegFormer_B4 | [segformer_b4_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b4_coco.py) | 60.8M/64.1M   | 58.5ms |  46.27               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b4/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b4/20220902_135723.log.json) |
| SegFormer_B5 | [segformer_b5_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/segformer/segformer_b5_coco.py) | 81.4M/85.7M   | 99.2ms |  46.75               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b5/SegmentationEvaluator_mIoU_best.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/damo/modelzoo/segmentation/segformer/segformer_b5/20220812_144336.log.json) |
