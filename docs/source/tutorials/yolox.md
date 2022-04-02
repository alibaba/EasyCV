# yolox tutorial

## Data preparation
To download the dataset, please refer to [prepare_data.md](../prepare_data.md).

Yolox support both coco format and [PAI-Itag detection format](https://help.aliyun.com/document_detail/311173.html#title-y6p-ger-5l7),

### COCO format
To use coco data to train detection, you can refer to [configs/detection/yolox/yolox_s_8xb16_300e_coco.py](../../configs/detection/yolox/yolox_s_8xb16_300e_coco.py) for more configuration details.

### PAI-Itag detection format
To use pai-itag detection format data to train detection, you can refer to [configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py](../../configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py) for more configuration details.

## Local & PAI-DSW

To use COCO format data, use config file `configs/detection/yolox/yolox_s_8xb16_300e_coco.py`

To use PAI-Itag format data, use config file `configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py`


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
bash tools/dist_train.sh configs/detection/yolox/yolox_s_8xb16_300e_coco.py $GPUS
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
bash tools/dist_test.sh configs/detection/yolox/yolox_s_8xb16_300e_coco.py $GPUS work_dirs/detection/yolox/epoch_300.pth --eval
```

### Export model

```shell
python tools/export.py \
		${CONFIG_PATH} \
		${CHECKPOINT} \
		${EXPORT_PATH}
```

<details>
<summary>Arguments</summary>

- `CONFIG_PATH`: the config file path of a detection method
- `CHECKPOINT`:your checkpoint file of a detection method named as epoch_*.pth.
- `EXPORT_PATH`: your path to save export model

</details>

**Examples:**

```shell
python tools/export.py configs/detection/yolox/yolox_s_8xb16_300e_coco.py \
        work_dirs/detection/yolox/epoch_300.pth \
        work_dirs/detection/yolox/epoch_300_export.pth
```

### Inference
Download [test_image](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/small_coco_demo/val2017/000000017627.jpg)

```python
import cv2
from easycv.predictors import TorchYoloXPredictor

output_ckpt = 'work_dirs/detection/yolox/epoch_300.pth'
detector = TorchYoloXPredictor(output_ckpt)

img = cv2.imread('000000017627.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output = detector.predict([img])
print(output)

# visualize image
from matplotlib import pyplot as plt
image = img.copy()
for box, cls_name in zip(output[0]['detection_boxes'], output[0]['detection_class_names']):
    # box is [x1,y1,x2,y2]
    box = [int(b) for b in box]
    image = cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), (0,255,0), 2)
    cv2.putText(image, cls_name, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
plt.imshow(image)
plt.show()
```
