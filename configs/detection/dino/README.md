# DINO

> [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

<!-- [ALGORITHM] -->

## Abstract

We present DINO(DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box pre- diction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yield- ing a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.

<div align=center>
<img src="https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/algo_images/detection/DINO.png"/>
</div>

## Results and Models

| Algorithm  | Config                                                       | Params<br/>(backbone/total)                            | inference time(V100)<br/>(ms/img)                      | bbox_mAP<sup>val<br/><sub>0.5:0.95</sub> | AP<sup>val<br/><sub>50</sub> | Download                                                     |    Comment         |
| ---------- | ------------------------------------------------------------ | ------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| DINO_4sc_r50_12e    | [DINO_4sc_r50_12e](https://github.com/alibaba/EasyCV/tree/master/configs/detection/dino/dino_4sc_r50_12e_coco.py) | 23M/47M | 184ms |     48.71               |     66.27      | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_r50_12e/epoch_12.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_r50_12e/20220815_141403.log.json) |推理使用V100 32G|
| DINO_4sc_r50_36e    | [DINO_4sc_r50_36e](https://github.com/alibaba/EasyCV/tree/master/configs/detection/dino/dino_4sc_r50_36e_coco.py) | 23M/47M | 184ms |        50.69            |     68.60      | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_r50_36e/epoch_29.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_r50_36e/20220817_101549.log.json) |推理使用V100 32G|
| DINO_4sc_swinl_12e    | [DINO_4sc_swinl_12e](https://github.com/alibaba/EasyCV/tree/master/configs/detection/dino/dino_4sc_swinl_12e_coco.py) | 195M/217M | 155ms |        56.86            |     75.61      | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_swinl_12e/epoch_12.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_swinl_12e/20220815_211633.log.json) |推理使用V100 32G|
| DINO_4sc_swinl_36e    | [DINO_4sc_swinl_36e](https://github.com/alibaba/EasyCV/tree/master/configs/detection/dino/dino_4sc_swinl_36e_coco.py) | 195M/217M | 155ms |          58.04          |     76.76      | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_swinl_36e/epoch_34.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_swinl_36e/20220817_101416.log.json) |推理使用V100 32G|
| DINO_5sc_swinl_36e    | [DINO_5sc_swinl_36e](https://github.com/alibaba/EasyCV/tree/master/configs/detection/dino/dino_5sc_swinl_36e_coco.py) | 195M/217M | 235ms |           58.47         |     77.10      | [model](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_5sc_swinl_36e/epoch_35.pth) - [log](https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_5sc_swinl_36e/20220820_215711.log.json) |推理使用V100 32G|

## Citation

```latex
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
