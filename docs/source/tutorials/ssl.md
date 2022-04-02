# self-supervised learning tutorial


## Data Preparation
To download the dataset, please refer to [prepare_data.md](../prepare_data.md).

Self-supervised learning support imagenet(raw and tfrecord) format data.

### Imagenet format
You can download Imagenet data or use your own unlabeld image data. You should provide a directory which contains images for self-supervised training and a filelist which contains image path to the root directory.  For example, the image directory is as follows
```
images/
├── 0001.jpg
├── 0002.jpg
├── 0003.jpg
|...
└── 9999.jpg
```

the content of filelist is
```text
0001.jpg
0002.jpg
0003.jpg
...
9999.jpg
```


## Local & PAI-DSW

We use [configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_jpg.py](../../configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_jpg.py) as an example config in which two config variable should be modified

```python
data_train_list = 'filelist.txt'
data_train_root = 'images'
```


### Training

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

- `CONFIG_PATH`: the config file path of a selfsup method

- `WORK_DIR`: your path to save models and logs

</details>

**Examples:**

Edit `data_root`path in the `${CONFIG_PATH}` to your own data path.

```shell
GPUS=8
bash tools/dist_train.sh configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_jpg.py $GPUS
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

- `CONFIG_PATH`: the config file path of a selfsup method
- `CHECKPOINT`:your checkpoint file of a selfsup method named as epoch_*.pth
- `EXPORT_PATH`: your path to save export model

</details>

**Examples:**

```shell
python tools/export.py configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_jpg.py \
    work_dirs/selfsup/mocov2/epoch_200.pth \
    work_dirs/selfsup/mocov2/epoch_200_export.pth
```

### Feature extract
Download [test_image](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/product_detection/248347732153_1040.jpg)

```python
import cv2
from easycv.predictors.feature_extractor import TorchFeatureExtractor

output_ckpt = 'work_dirs/selfsup/mocov2/epoch_200_export.pth'
fe = TorchFeatureExtractor(output_ckpt)

img = cv2.imread('248347732153_1040.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
feature = fe.predict([img])
print(feature[0]['feature'].shape)
```
