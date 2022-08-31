
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/pai-easycv)](https://pypi.org/project/pai-easycv/)
[![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/alibaba/EasyCV.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/alibaba/EasyCV.svg)](https://github.com/alibaba/EasyCV/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/alibaba/EasyCV.svg)](https://GitHub.com/alibaba/EasyCV/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/alibaba/EasyCV)](https://GitHub.com/alibaba/EasyCV/commit/)
<!-- [![GitHub contributors](https://img.shields.io/github/contributors/alibaba/EasyCV.svg)](https://GitHub.com/alibaba/EasyCV/graphs/contributors/) -->
<!-- [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) -->


</div>


# EasyCV

English | [简体中文](README_zh-CN.md)

## Introduction

EasyCV is an all-in-one computer vision toolbox based on PyTorch, mainly focuses on self-supervised learning, transformer based models, and major CV tasks including image classification, metric-learning, object detection, pose estimation and so on.

[🔥 Latest News] We have released our YOLOX-PAI that achieves SOTA results within 40~50 mAP (less than 1ms). And we also provide a convenient and fast export/predictor api for end2end object detection. To get a quick start of YOLOX-PAI, click [here](docs/source/tutorials/yolox.md)!

### Major features

- **SOTA SSL Algorithms**

  EasyCV provides state-of-the-art algorithms in self-supervised learning based on contrastive learning such as SimCLR, MoCO V2, Swav, DINO and also MAE based on masked image modeling. We also provide standard benchmarking tools for ssl model evaluation.

- **Vision Transformers**

  EasyCV aims to provide an easy way to use the off-the-shelf SOTA transformer models trained either using supervised learning or self-supervised learning, such as ViT, Swin Transformer and DETR Series. More models will be added in the future. In addition, we support all the pretrained models from [timm](https://github.com/rwightman/pytorch-image-models).

- **Functionality & Extensibility**

  In addition to SSL, EasyCV also supports image classification, object detection, metric learning, and more areas will be supported in the future. Although covering different areas,
  EasyCV decomposes the framework into different components such as dataset, model and running hook, making it easy to add new components and combining it with existing modules.

  EasyCV provides simple and comprehensive interface for inference. Additionally,  all models are supported on [PAI-EAS](https://help.aliyun.com/document_detail/113696.html), which can be easily deployed as online service and support automatic scaling and service monitoring.

- **Efficiency**

  EasyCV supports multi-gpu and multi worker training. EasyCV uses [DALI](https://github.com/NVIDIA/DALI) to accelerate data io and preprocessing process, and uses [TorchAccelerator](https://github.com/alibaba/EasyCV/tree/master/docs/source/tutorials/torchacc.md) and fp16 to accelerate training process. For inference optimization, EasyCV exports model using jit script, which can be optimized by [PAI-Blade](https://help.aliyun.com/document_detail/205134.html)

## Technical Articles

We have a series of technical articles on the functionalities of EasyCV.
* [EasyCV开源｜开箱即用的视觉自监督+Transformer算法库](https://zhuanlan.zhihu.com/p/505219993)
* [MAE自监督算法介绍和基于EasyCV的复现](https://zhuanlan.zhihu.com/p/515859470)
* [基于EasyCV复现ViTDet：单层特征超越FPN](https://zhuanlan.zhihu.com/p/528733299)
* [基于EasyCV复现DETR和DAB-DETR，Object Query的正确打开方式](https://zhuanlan.zhihu.com/p/543129581)

## Installation

Please refer to the installation section in [quick_start.md](docs/source/quick_start.md) for installation.


## Get Started

Please refer to [quick_start.md](docs/source/quick_start.md) for quick start. We also provides tutorials for more usages.

* [self-supervised learning](docs/source/tutorials/ssl.md)
* [image classification](docs/source/tutorials/cls.md)
* [object detection with yolox-pai](docs/source/tutorials/yolox.md)
* [model compression with yolox](docs/source/tutorials/compression.md)
* [metric learning](docs/source/tutorials/metric_learning.md)
* [torchacc](docs/source/tutorials/torchacc.md)

notebook
* [self-supervised learning](docs/source/tutorials/EasyCV图像自监督训练-MAE.ipynb)
* [image classification](docs/source/tutorials/EasyCV图像分类resnet50.ipynb)
* [object detection with yolox-pai](docs/source/tutorials/EasyCV图像检测YoloX.ipynb)
* [metric learning](docs/source/tutorials/EasyCV度量学习resnet50.ipynb)


## Model Zoo

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center">
      <td>
        <b>Self-Supervised Learning</b>
      </td>
      <td>
        <b>Image Classification</b>
      </td>
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Segmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/selfsup/byol">BYOL (NeurIPS'2020)</a></li>
            <li><a href="configs/selfsup/dino">DINO (ICCV'2021)</a></li>
            <li><a href="configs/selfsup/mixco">MiXCo (NeurIPS'2020)</a></li>
            <li><a href="configs/selfsup/moby">MoBY (ArXiv'2021)</a></li>
            <li><a href="configs/selfsup/mocov2">MoCov2 (ArXiv'2020)</a></li>
            <li><a href="configs/selfsup/simclr">SimCLR (ICML'2020)</a></li>
            <li><a href="configs/selfsup/swav">SwAV (NeurIPS'2020)</a></li>
            <li><a href="configs/selfsup/mae">MAE (CVPR'2022)</a></li>
            <li><a href="configs/selfsup/fast_convmae">FastConvMAE (ArXiv'2022)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/classification/imagenet/resnet">ResNet (CVPR'2016)</a></li>
          <li><a href="configs/classification/imagenet/resnext">ResNeXt (CVPR'2017)</a></li>
          <li><a href="configs/classification/imagenet/hrnet">HRNet (CVPR'2019)</a></li>
          <li><a href="configs/classification/imagenet/vit">ViT (ICLR'2021)</a></li>
          <li><a href="configs/classification/imagenet/swint">SwinT (ICCV'2021)</a></li>
          <li><a href="configs/classification/imagenet/efficientformer">EfficientFormer (ArXiv'2022)</a></li>
          <li><a href="configs/classification/imagenet/timm/deit">DeiT (ICML'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/xcit">XCiT (ArXiv'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/tnt">TNT (NeurIPS'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/convit">ConViT (ArXiv'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/cait">CaiT (ICCV'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/levit">LeViT (ICCV'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/convnext">ConvNeXt (CVPR'2022)</a></li>
          <li><a href="configs/classification/imagenet/timm/resmlp">ResMLP (ArXiv'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/coat">CoaT (ICCV'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/convmixer">ConvMixer (ICLR'2022)</a></li>
          <li><a href="configs/classification/imagenet/timm/mlp-mixer">MLP-Mixer (ArXiv'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/nest">NesT (AAAI'2022)</a></li>
          <li><a href="configs/classification/imagenet/timm/pit">PiT (ArXiv'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/twins">Twins (NeurIPS'2021)</a></li>
          <li><a href="configs/classification/imagenet/timm/shuffle_transformer">Shuffle Transformer (ArXiv'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/detection/fcos">FCOS (ICCV'2019)</a></li>
          <li><a href="configs/detection/yolox">YOLOX (ArXiv'2021)</a></li>
          <li><a href="configs/detection/yolox">YOLOX-PAI (ArXiv'2022)</a></li>
          <li><a href="configs/detection/detr">DETR (ECCV'2020)</a></li>
          <li><a href="configs/detection/dab_detr">DAB-DETR (ICLR'2022)</a></li>
          <li><a href="configs/detection/dab_detr">DN-DETR (CVPR'2022)</a></li>
          <li><a href="configs/detection/dino">DINO (ArXiv'2022)</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Instance Segmentation</b></li>
        <ul>
        <ul>
          <li><a href="configs/detection/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/detection/vitdet">ViTDet (ArXiv'2022)</a></li>
          <li><a href="configs/segmentation/mask2former">Mask2Former (CVPR'2022)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Sementic Segmentation</b></li>
        <ul>
        <ul>
          <li><a href="configs/segmentation/fcn">FCN (CVPR'2015)</a></li>
          <li><a href="configs/segmentation/upernet">UperNet (ECCV'2018)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Panoptic Segmentation</b></li>
        <ul>
        <ul>
          <li><a href="configs/segmentation/mask2former">Mask2Former (CVPR'2022)</a></li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>


Please refer to the following model zoo for more details.

- [self-supervised learning model zoo](docs/source/model_zoo_ssl.md)
- [classification model zoo](docs/source/model_zoo_cls.md)
- [detection model zoo](docs/source/model_zoo_det.md)
- [segmentation model zoo](docs/source/model_zoo_seg.md)

## Data Hub

EasyCV have collected dataset info for different senarios, making it easy for users to fintune or evaluate models in EasyCV modelzoo.

Please refer to [data_hub.md](docs/source/data_hub.md).

## ChangeLog

* 31/08/2022 EasyCV v0.6.0 was released.
  -  Release YOLOX-PAI which achieves SOTA results within 40~50 mAP (less than 1ms) (#154 #172  #174 )
  -  Add detection algo DINO (#144)
  -  Add mask2former algo (#115)
  -  Releases imagenet1k, imagenet22k, coco, lvis, voc2012 data with BaiduDisk to accelerate downloading (#145 )

* 28/07/2022 EasyCV v0.5.0 was released.

* 23/06/2022 EasyCV v0.4.0 was released.

* 31/04/2022 EasyCV v0.3.0 was released.

* 07/04/2022 EasyCV v0.2.2 was released.

Please refer to [change_log.md](docs/source/change_log.md) for more details and history.


## License

This project licensed under the [Apache License (Version 2.0)](LICENSE). This toolkit also contains various third-party components and some code modified from other repos under other open source licenses. See the [NOTICE](NOTICE) file for more information.


## Contact

This repo is currently maintained by PAI-CV team, you can contact us by
* Dingding group number: 41783266
* Email: easycv@list.alibaba-inc.com

### Enterprise Service
If you need EasyCV enterprise service support, or purchase cloud product services, you can contact us by DingDing Group.

![dingding_qrcode](https://user-images.githubusercontent.com/4771825/165244727-b5d69628-97a6-4e2a-a23f-0c38a8d29341.jpg)
