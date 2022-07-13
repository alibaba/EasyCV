# Segmentation Model Zoo

## FCN

Pretrained on **Pascal VOC 2012 + Aug**.

| Algorithm  | Config                                                       | mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| fcn_r50_d8 | [fcn_r50-d8_512x512_8xb4_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/fcn/fcn_r50-d8_512x512_8xb4_60e_voc12aug.py) | 69.01               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/fcn_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/fcn_r50/20220525_203606.log.json) |

## UperNet

Pretrained on **Pascal VOC 2012 + Aug**.
| Algorithm  | Config                                                       | mIoU | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| upernet_r50 | [upernet_r50_512x512_8xb4_60e_voc12aug](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/upernet/upernet_r50_512x512_8xb4_60e_voc12aug.py) | 76.59               | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/upernet_r50/epoch_60.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/upernet_r50/20220706_114712.log.json) |

## Mask2former

### Instance Segmentation on COCO
| Algorithm  | Config                                                       | box MAP | Mask mAP | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ |----------|---------------------------------------------------------------------------- |
| mask2former_r50 | [mask2former_r50_8xb2_e50_instance](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/mask2former/mask2former_r50_8xb2_e50_instance.py) | 46.09 | 43.26 |[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_instance/epoch_50.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_instance/20220620_113639.log.json) |

### Panoptic Segmentation on COCO
| Algorithm  | Config                                                       | PQ | box MAP | Mask mAP | Download                                                     |
| ---------- | ---------- | ------------------------------------------------------------ | ------------------------ |----------|---------------------------------------------------------------------------- |
| mask2former_r50 | [mask2former_r50_8xb2_e50_panopatic](https://github.com/alibaba/EasyCV/tree/master/configs/segmentation/mask2former/mask2former_r50_8xb2_e50_panopatic.py) | 51.64 | 44.81 | 41.88 |[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_panoptic/epoch_50.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/mask2former_r50_panoptic/20220629_170721.log.json) |
