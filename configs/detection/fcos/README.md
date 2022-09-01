# FCOS

> [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

<!-- [ALGORITHM] -->

## Abstract

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks.

<div align=center>
<img src="https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/algo_images/detection/fcos.png"/>
</div>

## Results and Models

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | inference time(V100)<br/>(ms/img)                      | mAP<sup>val<br/><sub>0.5:0.95</sub> | AP<sup>val<br/><sub>50</sub> | Download                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FCOS-r50(caffe)    | [fcos-r50](https://github.com/alibaba/EasyCV/tree/master/configs/detection/fcos/fcos_r50_caffe_1x_coco.py) | 23M/32M | 85.8ms | 38.58                   | 57.18          | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/epoch_12.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/20220621_121315.log.json) |
| FCOS-r50(torch)    | [fcos-r50](https://github.com/alibaba/EasyCV/tree/master/configs/detection/fcos/fcos_r50_torch_1x_coco.py) | 23M/32M | 105.3ms | 38.88                   | 58.01          | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/fcos_epoch_12.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/20220826_182628.log.json) |

## Citation

```latex
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```
