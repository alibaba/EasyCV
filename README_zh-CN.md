
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

[English](README.md) | 简体中文

## 简介

EasyCV是一个涵盖多个领域的基于Pytorch的计算机视觉工具箱，聚焦自监督学习和视觉transformer关键技术，覆盖主流的视觉建模任务例如图像分类，度量学习，目标检测，关键点检测等。

### 核心特性

- **SOTA 自监督算法**

  EasyCV提供了state-of-the-art的自监督算法，有基于对比学习的算法例如 SimCLR，MoCO V2，Swav， Moby，DINO，也有基于掩码图像建模的MAE算法，除此之外我们还提供了标准的benchmark工具用来进行自监督算法模型的效果评估。

- **视觉Transformers**

  EasyCV聚焦视觉transformer技术，希望通过一种简洁的方式让用户方便地使用各种SOTA的、基于自监督预训练和imagenet预训练的视觉transformer模型，例如ViT，Swin-Transformer，Shuffle Transformer，未来也会加入更多相关模型。此外，我们还支持所有[timm](https://github.com/rwightman/pytorch-image-models)仓库中的预训练模型.

- **易用性和可扩展性**

  除了自监督学习，EasyCV还支持图像分类、目标检测，度量学习，关键点检测等领域，同时未来也会支持更多任务领域。 尽管横跨多个任务领域，EasyCV保持了统一的架构，整体框架划分为数据集、模型、回调模块，非常容易增加新的算法、功能，以及基于现有模块进行扩展。

  推理方面，EasyCV提供了端到端的简单易用的推理接口，支持上述多个领域。 此外所有的模型都支持使用[PAI-EAS](https://help.aliyun.com/document_detail/113696.html)进行在线部署，支持自动伸缩和服务监控。

- **高性能**

  EasyCV支持多机多卡训练，同时支持[TorchAccelerator](docs/source/tutorials/torchacc.md)和fp16进行训练加速。在数据读取和预处理方面，EasyCV使用[DALI](https://github.com/NVIDIA/DALI)进行加速。对于模型推理优化，EasyCV支持使用jit script导出模型，使用[PAI-Blade](https://help.aliyun.com/document_detail/205134.html)进行模型优化。

## 技术文章

我们有一系列关于EasyCV功能的技术文章。
* [EasyCV开源｜开箱即用的视觉自监督+Transformer算法库](https://zhuanlan.zhihu.com/p/505219993)
* [MAE自监督算法介绍和基于EasyCV的复现](https://zhuanlan.zhihu.com/p/515859470)
* [基于EasyCV复现ViTDet：单层特征超越FPN](https://zhuanlan.zhihu.com/p/528733299)
* [基于EasyCV复现DETR和DAB-DETR，Object Query的正确打开方式](https://zhuanlan.zhihu.com/p/543129581)

## 安装

请参考[快速开始教程](docs/source/quick_start.md)中的安装章节。


## 快速开始

请参考[快速开始教程](docs/source/quick_start.md) 快速开始。我们也提供了更多的教程方便你的学习和使用。

* [自监督学习教程](docs/source/tutorials/ssl.md)
* [图像分类教程](docs/source/tutorials/cls.md)
* [使用YOLOX进行物体检测教程](docs/source/tutorials/yolox.md)
* [YOLOX模型压缩教程](docs/source/tutorials/compression.md)
* [torchacc](docs/source/tutorials/torchacc.md)

## 模型库

不同领域的模型仓库和benchmark指标如下

- [自监督模型库](docs/source/model_zoo_ssl.md)
- [图像分类模型库](docs/source/model_zoo_cls.md)
- [目标检测模型库](docs/source/model_zoo_det.md)


## 变更日志

* 28/07/2022 EasyCV v0.5.0 版本发布。
    * 自监督学习增加了ConvMAE算法
    * 图像分类增加EfficientFormer
    * 目标检测增加FCOS、DETR、DAB-DETR和DN-DETR算法
    * 语义分割增加了UperNet算法
    * 支持使用[torchacc](docs/source/tutorials/torchacc.md)加快训练速度
    * 增加模型分析工具

* 23/06/2022 EasyCV v0.4.0 版本发布。
    * 增加语义分割模块， 支持FCN算法
    * 扩充分类算法 model zoo
    * Yolox支持导出 [blade](https://help.aliyun.com/document_detail/205134.html) 模型
    * 支持 ViTDet 检测算法
    * 支持 sailfish 数据并行训练
    * 支持运行 [mmdetection](https://github.com/open-mmlab/mmdetection) 中的模型

* 31/04/2022 EasyCV v0.3.0 版本发布。
    * 增加 moby deit-small 预训练模型
    * 增加 mae vit-large benchmark和预训练模型
    * 支持 tensorboard和wandb 的图像可视化

* 2022/04/07 EasyCV v0.2.2 版本发布。

更多详细变更日志请参考[变更记录](docs/source/change_log.md)。


## 开源许可证

本项目使用 [Apache 2.0 开源许可证](LICENSE). 项目内含有一些第三方依赖库源码，部分实现借鉴其他开源仓库，仓库名称和开源许可证说明请参考[NOTICE文件](NOTICE)。


## Contact

本项目由阿里云机器学习平台PAI-CV团队维护，你可以通过如下方式联系我们：

钉钉群号: 41783266
邮箱: easycv@list.alibaba-inc.com

### 企业级服务

如果你需要针对EasyCV提供企业级服务，或者购买云产品服务，你可以通过加入钉钉群联系我们。

![dingding_qrcode](https://user-images.githubusercontent.com/4771825/165244727-b5d69628-97a6-4e2a-a23f-0c38a8d29341.jpg)
