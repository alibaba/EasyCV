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
