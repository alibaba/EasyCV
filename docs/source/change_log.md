# v 0.10.0 (06/03/2023)

## Highlights
- Support STDC, STGCN, ReID and Multi-len MOT.
- Support multi processes for predictor data preprocessing. For the model with more time consuming in data preprocessing, the speedup can reach more than 50%.

## New Features
- Support multi processes for predictor data preprocessing. ([#272](https://github.com/alibaba/EasyCV/pull/272))
- Support STDC model. ([#284](https://github.com/alibaba/EasyCV/pull/284)) ([#286](https://github.com/alibaba/EasyCV/pull/286))
- Support ReID and Multi-len MOT. ([#285](https://github.com/alibaba/EasyCV/pull/285)) ([#295](https://github.com/alibaba/EasyCV/pull/295))
- Support STGCN model, and support export blade model. ([#293](https://github.com/alibaba/EasyCV/pull/293)) ([#299](https://github.com/alibaba/EasyCV/pull/299))
- Add pose model zoo and support export torch jit and blade model for pose models. ([#294](https://github.com/alibaba/EasyCV/pull/294))
- Support train motchallenge and crowdhuman datasets for detection models. ([#265](https://github.com/alibaba/EasyCV/pull/265))

## Improvements
- Speed up inference for face detector when using mtcnn. ([#273](https://github.com/alibaba/EasyCV/pull/273))
- Add mobilenet config for [itag](https://help.aliyun.com/document_detail/311162.html) and imagenet dataset, and optimize `ClsSourceImageList` api to support string label. ([#276](https://github.com/alibaba/EasyCV/pull/276)) ([#283](https://github.com/alibaba/EasyCV/pull/283))
- Support multi-rows replacement for first order parameter. ([#282](https://github.com/alibaba/EasyCV/pull/282))
- Add a tool to convert itag dataset to raw dataset. ([#290](https://github.com/alibaba/EasyCV/pull/290))
- Add `PoseTopDownPredictor` to replace `TorchPoseTopDownPredictorWithDetector` ([#296](https://github.com/alibaba/EasyCV/pull/296))

## Bug Fixes
- Remove git lfs dependencies. ([#278](https://github.com/alibaba/EasyCV/pull/278))
- Fix wholebody keypoints evaluation. ([#287](https://github.com/alibaba/EasyCV/pull/287))
- Fix DetSourceRaw while label file and image file not match. ([#289](https://github.com/alibaba/EasyCV/pull/289))

# v 0.9.0 (17/01/2023)

## Highlights
- Support Single-lens MOT ([#258](https://github.com/alibaba/EasyCV/pull/258))
- Support video recognition (X3D, SWIN-video) ([#256](https://github.com/alibaba/EasyCV/pull/256))

## New Features
- Add inception config and voc config for FCN and UperNet ([#261](https://github.com/alibaba/EasyCV/pull/261))
- Add inference time under V100 for the benchmark of deitiii and hydra attention ([#251](https://github.com/alibaba/EasyCV/pull/251))
- Add bev-blancehybrid benchmark ([#249](https://github.com/alibaba/EasyCV/pull/249))

## Improvements
- Optimize data source apis ([#254](https://github.com/alibaba/EasyCV/pull/254))
- Update predict.py to support input model directory ([#252](https://github.com/alibaba/EasyCV/pull/252))


## Bug Fixes
- Fix MAE arg error after timm upgrade ([#255](https://github.com/alibaba/EasyCV/pull/255))
- Fix export SSL models bug, avoid loading default pretrained backbone model ([#257](https://github.com/alibaba/EasyCV/pull/257))
- Fix bug can't find config files while easycv is installed ([#253](https://github.com/alibaba/EasyCV/pull/253))

# v 0.8.0 (5/12/2022)

## Highlights
- Add BEVFormer and improve the performance of BEVFormer ([#224](https://github.com/alibaba/EasyCV/pull/224))
- Add DINO++ and support objects365 pretrain ([#242](https://github.com/alibaba/EasyCV/pull/242))

## New Features
- Add DeiT of Hydra Attention version ([#220](https://github.com/alibaba/EasyCV/pull/220))
- Add EdgeViTv3 ([#214](https://github.com/alibaba/EasyCV/pull/214))
- Add BEVFormer and improve the performance of BEVFormer ([#224](https://github.com/alibaba/EasyCV/pull/224))
- Add DINO++ and support objects365 pretrain ([#242](https://github.com/alibaba/EasyCV/pull/242))

## Improvements
- Unify the parsing method of config scripts, and support both local and pai platform products ([#235](https://github.com/alibaba/EasyCV/pull/235))
- Add more data source apis for open source datasets, involving classification, detection, segmentation and keypoints tasks. And part of the data source apis support automatic download. For more information, please refer to [data_hub](https://github.com/alibaba/EasyCV/blob/master/docs/source/data_hub.md) ([#206](https://github.com/alibaba/EasyCV/pull/206) [#229](https://github.com/alibaba/EasyCV/pull/229))
- Add confusion matrix metric for Classification models ([#241](https://github.com/alibaba/EasyCV/pull/241))
- Add prediction script ([#239](https://github.com/alibaba/EasyCV/pull/239))

## Bug Fixes
- Sync the predict config in the config file for predictor ([#238](https://github.com/alibaba/EasyCV/pull/238))
- Fix index of image_scale with y2 with bottom_left implemented in _mosaic_combine ([#231](https://github.com/alibaba/EasyCV/pull/231))
- Add bevformer benchmark and fix classification predict bug ([#240](https://github.com/alibaba/EasyCV/pull/240))

# v 0.7.0 (3/11/2022)

## Highlights
- Support auto hyperparameter optimization of NNI ([#211](https://github.com/alibaba/EasyCV/pull/211))
- Add DeiT III ([#171](https://github.com/alibaba/EasyCV/pull/171))
- Add semantic segmentation model SegFormer ([#191](https://github.com/alibaba/EasyCV/pull/191))
- Add 3d detection model BEVFormer ([#203](https://github.com/alibaba/EasyCV/pull/203))

## New Features
- Support semantic mask2former ([#199](https://github.com/alibaba/EasyCV/pull/199))
- Support face 2d keypoint detection ([#191](https://github.com/alibaba/EasyCV/pull/191))
- Support hand keypoints detection ([#191](https://github.com/alibaba/EasyCV/pull/191))
- Support wholebody keypoint detection ([#207](https://github.com/alibaba/EasyCV/pull/207))
- Support auto hyperparameter optimization of NNI ([#211](https://github.com/alibaba/EasyCV/pull/211))
- Add DeiT III ([#171](https://github.com/alibaba/EasyCV/pull/171))
- Add semantic segmentation model SegFormer ([#191](https://github.com/alibaba/EasyCV/pull/191))
- Add 3d detection model BEVFormer ([#203](https://github.com/alibaba/EasyCV/pull/203))

## Improvements
- Optimze predcitor apis, support cpu and batch inference ([#195](https://github.com/alibaba/EasyCV/pull/195))
- Speed up ViTDet model ([#177](https://github.com/alibaba/EasyCV/pull/177))
- Support export jit model end2end for yolox ([#215](https://github.com/alibaba/EasyCV/pull/215))

## Bug Fixes
- Fix the bug of io.copytree copying multiple directories ([#193](https://github.com/alibaba/EasyCV/pull/193))
- Move thirdparty into easycv ([#216](https://github.com/alibaba/EasyCV/pull/216))


# v 0.6.1 (06/09/2022)

## Bug Fixes

- Fix missing utils ([#183](https://github.com/alibaba/EasyCV/pull/183))

# v 0.6.0 (31/08/2022)

## Highlights
- Release YOLOX-PAI which achieves SOTA results within 40~50 mAP (less than 1ms) ([#154](https://github.com/alibaba/EasyCV/pull/154) [#172](https://github.com/alibaba/EasyCV/pull/172) [#174](https://github.com/alibaba/EasyCV/pull/174) )
- Add detection algo DINO ([#144](https://github.com/alibaba/EasyCV/pull/144))
- Add mask2former algo ([#115](https://github.com/alibaba/EasyCV/pull/115))
- Releases imagenet1k, imagenet22k, coco, lvis, voc2012 data with BaiduDisk to accelerate downloading ([#145](https://github.com/alibaba/EasyCV/pull/145) )

## New Features

- Add detection predictor which support model inference without exporting models([#158](https://github.com/alibaba/EasyCV/pull/158) )
- Add VitDet support for faster-rcnn ([#155](https://github.com/alibaba/EasyCV/pull/155) )
- Release YOLOX-PAI which achieves SOTA results within 40~50 mAP (less than 1ms) ([#154](https://github.com/alibaba/EasyCV/pull/154) [#172](https://github.com/alibaba/EasyCV/pull/172) [#174](https://github.com/alibaba/EasyCV/pull/174) )
- Support DINO algo ([#144](https://github.com/alibaba/EasyCV/pull/144))
- Add mask2former algo ([#115](https://github.com/alibaba/EasyCV/pull/115))

## Improvements

- FCOS update torch_style ([#170](https://github.com/alibaba/EasyCV/pull/170))
- Add algo tables to describe which algo EasyCV support ([#157](https://github.com/alibaba/EasyCV/pull/157) )
- Refactor datasources api ([#156](https://github.com/alibaba/EasyCV/pull/156) [#140](https://github.com/alibaba/EasyCV/pull/140) )
- Add PR and Issule template ([#150](https://github.com/alibaba/EasyCV/pull/150))
- Update Fast ConvMAE doc ([#151](https://github.com/alibaba/EasyCV/pull/151))

## Bug Fixes

- Fix YOLOXLrUpdaterHook conflict with mmdet ( [#169](https://github.com/alibaba/EasyCV/pull/169) )
- Fix datasource cache problem([#153](https://github.com/alibaba/EasyCV/pull/153))


# v 0.5.0 (28/07/2022)

## Highlights
- Self-Supervised support ConvMAE algorithm (([#101](https://github.com/alibaba/EasyCV/pull/101)) ([#121](https://github.com/alibaba/EasyCV/pull/121)))
- Classification support EfficientFormer algorithm ([#128](https://github.com/alibaba/EasyCV/pull/128))
- Detection support FCOS、DETR、DAB-DETR and DN-DETR algorithm (([#100](https://github.com/alibaba/EasyCV/pull/100)) ([#104](https://github.com/alibaba/EasyCV/pull/104)) ([#119](https://github.com/alibaba/EasyCV/pull/119)))
- Segmentation support UperNet algorithm ([#118](https://github.com/alibaba/EasyCV/pull/118))
- Support use torchacc to speed up training ([#105](https://github.com/alibaba/EasyCV/pull/105))

## New Features
- Support use analyze tools ([#133](https://github.com/alibaba/EasyCV/pull/133))

## Bug Fixes
- Update yolox config template and fix bugs ([#134](https://github.com/alibaba/EasyCV/pull/134))
- Fix yolox detector prediction export error ([#125](https://github.com/alibaba/EasyCV/pull/125))
- Fix common_io url error ([#126](https://github.com/alibaba/EasyCV/pull/126))

## Improvements
- Add ViTDet visualization ([#102](https://github.com/alibaba/EasyCV/pull/102))
- Refactor detection pipline ([#104](https://github.com/alibaba/EasyCV/pull/104))


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
