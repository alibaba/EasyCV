# DETR Turtorial

## Data preparation
To download the dataset, please refer to [prepare_data.md](../prepare_data.md).

### COCO format
To use coco data to train detection, you can refer to [configs/detection/detr/detr_r50_8x2_150e_coco.py](https://github.com/alibaba/EasyCV/tree/master/configs/detection/detr/detr_r50_8x2_150e_coco.py) for more configuration details.

## Get Started

To immediately use a model on a given input image, we provide the Predictor API. Predictor group together a pretrained model with the preprocessing that was used during that model's training. For example, we can easily extract detected objects in an image:

``` python
>>> from easycv.predictors.detector import DetectionPredictor

# Specify file path
>>> model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/detr/epoch_150.pth'
>>> config_path = 'configs/detection/detr/detr_r50_8x2_150e_coco.py'
>>> img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'

# Allocate a predictor for object detection
>>> model = DetectionPredictor(model_path, config_path)
>>> output = model(img)[0]
>>> model.visualize(img, output, out_file='./result.jpg')
output['detection_scores'][:2] = [0.58311516, 0.98532575]
output['detection_classes'][:2] = [2, 2]
output['detection_boxes'][:2] = [[1.32131638e+02, 9.08366165e+01, 1.51008240e+02, 1.01831055e+02],
								[1.89690186e+02, 1.08048561e+02, 2.96801422e+02, 1.54441940e+02]]

```

Here we get a list of objects detected in the image, with a box surrounding the object and a confidence score. The prediction results are as follows:


![result](../_static/result.jpg)

## Quick Start

To use COCO format data, use config file `configs/detection/detr/detr_r50_8x2_150e_coco.py`

You can use the [quick_start.md](../quick_start.md) for local installation or use our provided doker images.
```shell
registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:blade_cu111_easycv
```

### Train
**Single gpu:**

```shell
python tools/train.py \
		${CONFIG_PATH} \
		--work_dir ${WORK_DIR}
```

**Multi gpus:**

```shell
bash tools/dist_train.sh \
		${NUM_GPUS} \
		${CONFIG_PATH} \
		--work_dir ${WORK_DIR}
```

<details>
<summary>Arguments</summary>

- `NUM_GPUS`: number of gpus

- `CONFIG_PATH`: the config file path of a detection method

- `WORK_DIR`: your path to save models and logs

</details>

**Examples:**

Edit `data_root`path in the `${CONFIG_PATH}` to your own data path.

```shell
GPUS=8
bash tools/dist_train.sh configs/detection/detr/detr_r50_8x2_150e_coco.py $GPUS
```

### Evaluation

**Single gpu:**

```shell
python tools/eval.py \
		${CONFIG_PATH} \
		${CHECKPOINT} \
		--eval
```

**Multi gpus:**

```shell
bash tools/dist_test.sh \
		${CONFIG_PATH} \
		${NUM_GPUS} \
		${CHECKPOINT} \
		--eval
```

<details>
<summary>Arguments</summary>

- `CONFIG_PATH`: the config file path of a detection method

- `NUM_GPUS`: number of gpus

- `CHECKPOINT`: the checkpoint file named as epoch_*.pth.

</details>

**Examples:**

```shell
GPUS=8
bash tools/dist_test.sh configs/detection/detr/detr_r50_8x2_150e_coco.py $GPUS work_dirs/detection/detr/detr_150e.pth --eval
```

### Export model

```shell
python tools/export.py \
		${CONFIG_PATH} \
		${CHECKPOINT} \
		${EXPORT_PATH}
```

For more details of the export process, you can refer to [export.md](export.md).
<details>
<summary>Arguments</summary>

- `CONFIG_PATH`: the config file path of a detection method
- `CHECKPOINT`:your checkpoint file of a detection method named as epoch_*.pth.
- `EXPORT_PATH`: your path to save export model

</details>

**Examples:**

```shell
python tools/export.py configs/detection/detr/detr_r50_8x2_150e_coco.py \
        work_dirs/detection/detr/detr_150e.pth \
        work_dirs/detection/detr/detr_150e_export.pth
```
