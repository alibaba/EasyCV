
# EasyCV


## Introduction

EasyCV is an all-in-one computer vision toolbox based on PyTorch, mainly focus on self-supervised learning, transformer based models, and CV tasks including image classification, metric-learning, object detection and so on.

### Major features

- **SOTA SSL Algorithms**

  EasyCV provides state-of-the-art algorithms in self-supervised learning based on contrastive learning such as SimCLR, MoCO V2, Swav, DINO and also MAE based on masked image modeling. We also provides standard benchmark tools for ssl model evaluation.

- **Vision Transformers**

  EasyCV aims to provide plenty vision transformer models trained either using supervised learning or self-supervised learning, such as ViT, Swin-Transformer and XCit. More models will be added in the future.

- **Functionality & Extensibility**

  In addition to SSL, EasyCV also support image classification, object detection, metric learning, and more area will be supported in the future. Although convering different area,
  EasyCV decompose the framework into different componets such as dataset, model, running hook, making it easy to add new compoenets and combining it with existing modules.

  EasyCV provide simple and comprehensive interface for inference. Additionaly,  all models are supported on [PAI-EAS](https://help.aliyun.com/document_detail/113696.html), which can be easily deployed as online service and support automatic scaling and service moniting.

- **Efficiency**

  EasyCV support multi-gpu and multi worker training. EasyCV use [DALI](https://github.com/NVIDIA/DALI) to accelerate data io and preprocessing process, and use fp16 to accelerate training process. For inference optimization, EasyCV export model using jit script, which can be optimized by [PAI-Blade](https://help.aliyun.com/document_detail/205134.html)



## Installation

Please refer to the installation section in [quick_start.md](docs/source/quick_start.md) for installation.


## Get Started

Please refer to [quick_start.md](docs/source/quick_start.md) for quick start. We also provides tutorials for more usages.

* [self-supervised learning](docs/source/tutorials/ssl.md)
* [image classification](docs/source/tutorials/cls.md)
* [object detection with yolox](docs/source/tutorials/yolox.md)


## Model Zoo

Please refer to the following model zoo for more details.

- [self-supervised learning model zoo](docs/source/model_zoo_ssl.md)
- [detection model zoo](docs/source/model_zoo_detection.md)


## ChangeLog

* 07/04/2022 EasyCV v0.2.2 was released.

Please refer to [change_log.md](docs/source/change_log.md) for more details and history.


## License

This project licensed under the [Apache License (Version 2.0)](LICENSE). This toolkit also contains various third-party components and some code modified from other repos under other open source licenses. See the [NOTICE](NOTICE) file for more information.


## Contact

This repo is currently maintained by PAI-CV team, you can contact us by easycv@list.alibaba-inc.com or join the dingding group ([join url](https://h5.dingtalk.com/circle/healthCheckin.html?dtaction=os&corpId=ding3ff8258c1c5850ef6ef3dd1a991739ff&9abd5705-5633=d069c338-d566&cbdbhh=qwertyuiop)).

![dingding group QR code](docs/source/_static/dingding_qrcode.jpg)
