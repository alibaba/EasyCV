# Detection Model Zoo

## YOLOX

Pretrained on COCO2017 dataset.

| Algorithm  | Config                                                       | mAP<sup>val<br/><sub>0.5:0.95</sub> | AP<sup>val<br/><sub>50</sub> | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | --------------- | ------------------------------------------------------------ |
| YOLOX-s    | [yolox_s_8xb16_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_s_8xb16_300e_coco.py) | 40.0                   | 58.9          | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_s_bs16_lr002/epoch_300.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_s_bs16_lr002/log.txt) |
| YOLOX-m    | [yolox_m_8xb16_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_m_8xb16_300e_coco.py) | 46.3                   | 64.9          | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_m_bs16_lr002/epoch_300.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_m_bs16_lr002/log.txt) |
| YOLOX-l    | [yolox_l_8xb8_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_m_8xb8_300e_coco.py) | 48.9                  | 67.5        | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_l_bs8_lr001/epoch_290.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_l_bs8_lr001/log.txt) |
| YOLOX-x    | [yolox_x_8xb8_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_x_8xb8_300e_coco.py) | 50.9                   | 69.2          | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_x_bs8_lr001/epoch_290.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_x_bs8_lr001/log.txt) |
| YOLOX-tiny | [yolox_tiny_8xb16_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_tiny_8xb16_300e_coco.py) | 31.5                   | 49.2          | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_tiny_bs16_lr002/epoch_300.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_tiny_bs16_lr002/log.txt) |
| YOLOX-nano | [yolox_nano_8xb16_300e_coco](https://github.com/alibaba/EasyCV/tree/master/configs/detection/yolox/yolox_tiny_8xb16_300e_coco.py) | 26.5                   | 42.6          | [model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_nano_bs16_lr002/epoch_300.pth) - [log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox_nano_bs16_lr002/log.txt) |

## ViTDet

| Algorithm  | Config                                                       | bbox_mAP<sup>val<br/><sub>0.5:0.95</sub> | mask_mAP<sup>val<br/><sub>0.5:0.95</sub> | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | --------------- | ------------------------------------------------------------ |
| ViTDet_MaskRCNN    | [vitdet_maskrcnn](https://github.com/alibaba/EasyCV/tree/master/configs/detection/vitdet/vitdet_100e.py) | 50.57                   | 44.96          | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/vitdet/vit_base/vitdet_maskrcnn.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/vitdet/vit_base/vitdet_maskrcnn.log.json) |
