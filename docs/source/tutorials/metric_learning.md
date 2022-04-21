# metric learning tutorial

## Data Preparation
To download the dataset, please refer to [prepare_data.md](../prepare_data.md).

metric learning support cifar and imagenet(raw and tfrecord) format data.

### Imagenet format
You can also use your self-defined data which follows `imagenet format`, you should provide a root directory which condatains images for metric learning training and a filelist which contains image path to the root directory.  For example, the image root directory is as follows
```
images/
├── 0001.jpg
├── 0002.jpg
├── 0003.jpg
|...
└── 9999.jpg
```

each line of the filelist consists of two parts, subpath to the image files starting from the image root directory,  class label string for the corresponding image, which are seperated by space
```text
0001.jpg label1
0002.jpg label2
0003.jpg label3
...
9999.jpg label9999
```
To use Imagenet format data to train metric learning, you can refer to [configs/metric_learning/imagenet_resnet50_1000kid_jpg.py](../../configs/metric_learning/imagenet_resnet50_1000kid_jpg.py) for more configuration details.

## Local & PAI-DSW

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

- `CONFIG_PATH`: the config file path of a metric learning method

- `WORK_DIR`: your path to save models and logs

</details>

**Examples:**

Edit `data_root`path in the `${CONFIG_PATH}` to your own data path.

    single gpu training:
    ```shell
    python tools/train.py  configs/metric_learning/imagenet_resnet50_1000kid_jpg.py --work_dir work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg --fp16
    ```

    multi gpu training
    ```shell
    GPUS=8
    bash tools/dist_train.sh configs/metric_learning/imagenet_resnet50_1000kid_jpg.py $GPUS --fp16
    ```

    training using python api
    ```python
    import easycv.tools

    import os
    # config_path can be a local file or http url
    config_path = 'configs/metric_learning/imagenet_resnet50_1000kid_jpg.py'
    easycv.tools.train(config_path, gpus=8, fp16=False, master_port=29527)
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

- `CONFIG_PATH`: the config file path of a metric learning method

- `NUM_GPUS`: number of gpus

- `CHECKPOINT`: the checkpoint file named as epoch_*.pth

</details>

**Examples:**

    single gpu evaluation
    ```shell
    python tools/eval.py configs/metric_learning/imagenet_resnet50_1000kid_jpg.py work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350.pth --eval --fp16
    ```

    multi-gpu evaluation

    ```shell
    GPUS=8
    bash tools/dist_test.sh configs/metric_learning/imagenet_resnet50_1000kid_jpg $GPUS work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350.pth --eval --fp16
    ```

    evaluation using python api
    ```python
    import easycv.tools

    import os
    os.environ['CUDA_VISIBLE_DEVICES']='3,4,5,6'
    config_path = 'configs/metric_learning/imagenet_resnet50_1000kid_jpg'
    checkpoint_path = 'work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350.pth'
    easycv.tools.eval(config_path, checkpoint_path, gpus=8)
    ```

### Export model for inference
    If SyncBN is configured, we should replace it with BN in config file
    ```python
    # imagenet_resnet50_1000kid_jpg.py
    model = dict(
        ...
        backbone=dict(
            ...
            norm_cfg=dict(type='BN')), # SyncBN --> BN
        ...)
    ```

    ```shell
    python tools/export.py configs/metric_learning/imagenet_resnet50_1000kid_jpg \
        work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350.pth \
        work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350_export.pth
    ```

    or using python api
    ```python
    import easycv.tools

    config_path = './imagenet_resnet50_1000kid_jpg.py'
    checkpoint_path = 'oss://pai-vision-data-hz/pretrained_models/easycv/imagenet_resnet50_1000kid_jpg/resnet50.pth'
    export_path = './resnet50_export.pt'
    easycv.tools.export(config_path, checkpoint_path, export_path)
    ```

### Inference
    Download [test_image](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/cifar10/qince_data/predict/aeroplane_s_000004.png)


    ```python
    import cv2
    from easycv.predictors.classifier import TorchClassifier

    output_ckpt = 'work_dirs/metric_learning/imagenet_resnet50_1000kid_jpg/epoch_350_export.pth'
    tcls = TorchClassifier(output_ckpt)

    img = cv2.imread('aeroplane_s_000004.png')
    # input image should be RGB order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = tcls.predict([img])
    print(output)
    ```
