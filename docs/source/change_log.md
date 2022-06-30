# v 0.2.2 (07/04/2022)

* initial commit & first release

1. SOTA SSL Algorithms

EasyCV provides state-of-the-art algorithms in self-supervised learning based on contrastive learning such as SimCLR, MoCO V2, Swav, DINO and also MAE based on masked image modeling. We also provides standard benchmark tools for ssl model evaluation.

2. Vision Transformers

EasyCV aims to provide plenty vision transformer models trained either using supervised learning or self-supervised learning, such as ViT, Swin-Transformer and XCit. More models will be added in the future.

3. Functionality & Extensibility

In addition to SSL, EasyCV also support image classification, object detection, metric learning, and more area will be supported in the future. Although convering different area, EasyCV decompose the framework into different componets such as dataset, model, running hook, making it easy to add new compoenets and combining it with existing modules.
EasyCV provide simple and comprehensive interface for inference. Additionaly, all models are supported on PAI-EAS, which can be easily deployed as online service and support automatic scaling and service moniting.

3. Efficiency

EasyCV support multi-gpu and multi worker training. EasyCV use DALI to accelerate data io and preprocessing process, and use fp16 to accelerate training process. For inference optimization, EasyCV export model using jit script, which can be optimized by PAI-Blade.

# v 0.3.0 (05/05/2022)

## Highlights
- Support image visualization for tensorboard and wandb ([#15](https://github.com/alibaba/EasyCV/pull/15))

## New Features
- Update moby pretrained model to deit small ([#10](https://github.com/alibaba/EasyCV/pull/10))
- Support image visualization for tensorboard and wandb ([#15](https://github.com/alibaba/EasyCV/pull/15))
- Add mae vit-large benchmark  and pretrained models ([#24](https://github.com/alibaba/EasyCV/pull/24))

## Bug Fixes
-  Fix extract.py for benchmarks ([#7](https://github.com/alibaba/EasyCV/pull/7))
-  Fix inference error of classifier ([#19](https://github.com/alibaba/EasyCV/pull/19))
-  Fix multi-process reading of detection datasource and accelerate data preprocessing ([#23](https://github.com/alibaba/EasyCV/pull/23))
- Fix torchvision transforms wrapper ([#31](https://github.com/alibaba/EasyCV/pull/31))

## Improvements
- Add chinese readme ([#39](https://github.com/alibaba/EasyCV/pull/39))
- Add model compression tutorial ([#20](https://github.com/alibaba/EasyCV/pull/20))
- Add notebook tutorials ([#22](https://github.com/alibaba/EasyCV/pull/22))
- Uniform input and output format for transforms ([#6](https://github.com/alibaba/EasyCV/pull/6))
- Update model zoo link ([#8](https://github.com/alibaba/EasyCV/pull/8))
- Support readthedocs  ([#29](https://github.com/alibaba/EasyCV/pull/29))
- refine autorelease gitworkflow ([#13](https://github.com/alibaba/EasyCV/pull/13))

# v 0.4.0 (23/06/2022)

## Highlights
- Add **semantic segmentation** modules, support FCN algorithm ([#71](https://github.com/alibaba/EasyCV/pull/71))
- Expand classification model zoo ([#55](https://github.com/alibaba/EasyCV/pull/55))
- Support export model with **[blade](https://help.aliyun.com/document_detail/205134.html)** for yolox ([#66](https://github.com/alibaba/EasyCV/pull/66))
- Support **ViTDet** algorithm ([#35](https://github.com/alibaba/EasyCV/pull/35))
- Add sailfish for extensible fully sharded data parallel training ([#97](https://github.com/alibaba/EasyCV/pull/97))
- Support run with [mmdetection](https://github.com/open-mmlab/mmdetection) models ([#25](https://github.com/alibaba/EasyCV/pull/25))

## New Features
- Set multiprocess env for speedup ([#77](https://github.com/alibaba/EasyCV/pull/77))
- Add data hub, summarized various datasets in different fields ([#70](https://github.com/alibaba/EasyCV/pull/70))

## Bug Fixes
-  Fix the inaccurate accuracy caused by missing the `groundtruth_is_crowd` field in CocoMaskEvaluator ([#61](https://github.com/alibaba/EasyCV/pull/61))
- Unified the usage of `pretrained` parameter and fix load bugs（([#79](https://github.com/alibaba/EasyCV/pull/79)) ([#85](https://github.com/alibaba/EasyCV/pull/85)) ([#95](https://github.com/alibaba/EasyCV/pull/95))

## Improvements
- Update MAE pretrained models and benchmark ([#50](https://github.com/alibaba/EasyCV/pull/50))
- Add detection benchmark for SwAV and MoCo-v2 ([#58](https://github.com/alibaba/EasyCV/pull/58))
- Add moby swin-tiny pretrained model and benchmark ([#72](https://github.com/alibaba/EasyCV/pull/72))
- Update prepare_data.md, add more details ([#69](https://github.com/alibaba/EasyCV/pull/69))
- Optimize quantize code and support to export MNN model ([#44](https://github.com/alibaba/EasyCV/pull/44))
