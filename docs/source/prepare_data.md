# Prepare Datasets

- [Prepare Cifar](#Prepare-Cifar)
- [Prepare Imagenet](#Prepare-Imagenet)
- [Prepare Imagenet-TFrecords](#Prepare-Imagenet-TFrecords)
- [Prepare COCO](#Prepare-COCO)
- [Prepare PAI-Itag detection](#Prepare-PAI-Itag-detection)

## Prepare Cifar

Download dataset [cifar10](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/cifar10/cifar-10-python.tar.gz) and uncompress files to `data/cifar`,  directory structure is as follows:

```text
data/cifar
└── cifar-10-batches-py
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── readme.html
    ├── read.py
    └── test_batch
```

## Prepare Imagenet

1. Go to the [download-url](http://www.image-net.org/download-images), Register an account and log in .
2. Download the following files：

   - Training images (Task 1 & 2). 138GB.

   - Validation images (all tasks). 6.3GB.
3. Unzip the downloaded file.
4. Using this [scrip](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) to get data meta.

## Prepare Imagenet-TFrecords

1. Go to the [download-url](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0), Register an account and log in .
2. The dataset is divided into two parts, [part0](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0) (79GB) and [part1](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1) (75GB), you need download all of them.

## Prepare COCO

Download [COCO2017](https://cocodataset.org/#download) dataset to `data/coco`, directory structure is as follows

```text
data/coco
├── annotations
├── train2017
└── val2017
```

## Prepare PAI-Itag detection

Download [SmallCOCO](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/unittest/data/detection/small_coco_itag/small_coco_itag.tar.gz) dataset to `data/coco`,
directory structure is as follows:

```text
data/coco/
├── train2017
├── train2017_20_local.manifest
├── val2017
└── val2017_20_local.manifest
```

replace train_data and val_data path in config file
```shell
sed -i 's#train2017.manifest#train2017_20_local.manifest#g' configs/detection/yolox_coco_pai.py
sed -i 's#val2017.manifest#val2017_20_local.manifest#g' configs/detection/yolox_coco_pai.py
```
