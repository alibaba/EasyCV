
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

EasyCV is an all-in-one computer vision toolbox based on PyTorch, mainly focus on self-supervised learning, transformer based models, and SOTA CV tasks including image classification, metric-learning, object detection, pose estimation and so on.

### Major features

- **SOTA SSL Algorithms**

  EasyCV provides state-of-the-art algorithms in self-supervised learning based on contrastive learning such as SimCLR, MoCO V2, Swav, DINO and also MAE based on masked image modeling. We also provide standard benchmark tools for ssl model evaluation.

- **Vision Transformers**

  EasyCV aims to provide an easy way to use the off-the-shelf SOTA transformer models trained either using supervised learning or self-supervised learning, such as ViT, Swin-Transformer and Shuffle Transformer. More models will be added in the future. In addition, we support all the pretrained models from [timm](https://github.com/rwightman/pytorch-image-models).

- **Functionality & Extensibility**

  In addition to SSL, EasyCV also support image classification, object detection, metric learning, and more area will be supported in the future. Although convering different area,
  EasyCV decompose the framework into different componets such as dataset, model, running hook, making it easy to add new compoenets and combining it with existing modules.

  EasyCV provide simple and comprehensive interface for inference. Additionaly,  all models are supported on [PAI-EAS](https://help.aliyun.com/document_detail/113696.html), which can be easily deployed as online service and support automatic scaling and service monitoring.

- **Efficiency**

  EasyCV support multi-gpu and multi worker training. EasyCV use [DALI](https://github.com/NVIDIA/DALI) to accelerate data io and preprocessing process, and use fp16 to accelerate training process. For inference optimization, EasyCV export model using jit script, which can be optimized by [PAI-Blade](https://help.aliyun.com/document_detail/205134.html)



## Installation

Please refer to the installation section in [quick_start.md](docs/source/quick_start.md) for installation.


## Get Started

Please refer to [quick_start.md](docs/source/quick_start.md) for quick start. We also provides tutorials for more usages.

* [self-supervised learning](docs/source/tutorials/ssl.md)
* [image classification](docs/source/tutorials/cls.md)
* [object detection with yolox](docs/source/tutorials/yolox.md)
* [model compression with yolox](docs/source/tutorials/compression.md)
* [metric learning](docs/source/tutorials/metric_learning.md)

notebook
* [self-supervised learning](docs/source/tutorials/EasyCV图像自监督训练-MAE.ipynb)
* [image classification](docs/source/tutorials/EasyCV图像分类resnet50.ipynb)
* [object detection with yolox](docs/source/tutorials/EasyCV图像检测YoloX.ipynb)
* [metric learning](docs/source/tutorials/EasyCV度量学习resnet50.ipynb)


## Model Zoo

Please refer to the following model zoo for more details.

- [self-supervised learning model zoo](docs/source/model_zoo_ssl.md)
- [classification model zoo](docs/source/model_zoo_cls.md)
- [detection model zoo](docs/source/model_zoo_detection.md)
- [segmentation model zoo](docs/source/model_zoo_seg.md)

## Data Hub

EasyCV have collected dataset info for different senarios, making it easy for users to fintune or evaluate models in EasyCV modelzoo.

Please refer to [data_hub.md](https://github.com/alibaba/EasyCV/blob/master/docs/source/data_hub.md).

## ChangeLog

* 23/06/2022 EasyCV v0.4.0 was released.
    * Add semantic segmentation modules, support FCN algorithm
    * Expand classification model zoo
    * Support export model with [blade](https://help.aliyun.com/document_detail/205134.html) for yolox
    * Support ViTDet algorithm
    * Add sailfish for extensible fully sharded data parallel training
    * Support run with [mmdetection](https://github.com/open-mmlab/mmdetection) models

* 31/04/2022 EasyCV v0.3.0 was released.
    * Update moby pretrained model to deit small
    * Add mae vit-large benchmark and pretrained models
    * Support image visualization for tensorboard and wandb

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
