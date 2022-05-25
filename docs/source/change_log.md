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
