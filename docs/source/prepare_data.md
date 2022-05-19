# Prepare Datasets

EasyCV provides various datasets for multi tasks. Please refer to the following guide for data preparation and keep the same data structure.

- [Image Classification](#Image-Classification)
- [Object Detection](#Object-Detection)
- [Self-Supervised Learning](#Self-Supervised-Learning)
- [Pose (Keypoint)](#Pose-(Keypoint))

## Image Classification

- [Cifar10](#Cifar10)
- [Cifar100](#Cifar100)
- [Imagenet-1k](#Imagenet-1k)
- [Imagenet-1k-TFrecords](#Imagenet-1k-TFrecords)

### Cifar10

The CIFAR-10 are labeled subsets of the [80 million tiny images](http://people.csail.mit.edu/torralba/tinyimages/) dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

It consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

There are 50000 training images and 10000 test images.

Here is the list of classes in the CIFAR-10: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

For more detailed information, please refer to [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).

#### Download

Download data from  [cifar-10-python.tar.gz ](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)(163MB). And uncompress files to `data/cifar10`.

Directory structure is as follows:

```text
data/cifar10
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

### Cifar100

The CIFAR-100 are labeled subsets of the [80 million tiny images](http://people.csail.mit.edu/torralba/tinyimages/) dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each.

There are 500 training images and 100 testing images per class.

The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

For more detailed information, please refer to [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).

#### Download

Download data from [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) (161MB). And uncompress files to `data/cifar100`.

Directory structure should be as follows:

```text
data/cifar100
└── cifar-100-python
    ├── file.txt~
    ├── meta
    ├── test
    ├── train
```

### Imagenet-1k

ImageNet is an image database organized according to the [WordNet](http://wordnet.princeton.edu/) hierarchy (currently only the nouns).

It is used in the ImageNet Large Scale Visual Recognition Challenge(ILSVRC) and is a benchmark for image classification.

For more detailed information, please refer to [ImageNet](https://image-net.org/download.php).

#### Download

ILSVRC2012 is widely used, download it as follows:

1. Go to the [download-url](http://www.image-net.org/download-images), Register an account and log in .
2. Recommended ILSVRC2012, download the following files：

   - Training images (Task 1 & 2). 138GB.

   - Validation images (all tasks). 6.3GB.
3. Unzip the downloaded file.
4. Using this [scrip](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) to get data meta.

Directory structure should be  as follows:

```
data/imagenet
└── train
    └── n01440764
    └── n01443537
    └── ...
└── val
    └── n01440764
    └── n01443537
    └── ...
└── meta
    ├── train.txt
    ├── val.txt
    ├── ...
```

### Imagenet-1k-TFrecords

Original imagenet raw images packed in TFrecord format.

For more detailed information about Imagenet dataset, please refer to [ImageNet](https://image-net.org/download.php).

#### Download

1. Go to the [download-url](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0), Register an account and log in .
2. The dataset is divided into two parts, [part0](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0) (79GB) and [part1](https://www.kaggle.com/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1) (75GB), you need download all of them.

Directory structure should be  as follows, put the image file and the idx file in the same folder:

```
data/imagenet
└── train
    ├── train-00000-of-01024
    ├── train-00000-of-01024.idx
    ├── train-00001-of-01024
    ├── train-00001-of-01024.idx
    ├── ...
└── validation
    ├── validation-00000-of-00128
    ├── validation-00000-of-00128.idx
    ├── validation-00001-of-00128
    ├── validation-00001-of-00128.idx
    ├── ...
```

## Object Detection

- [PAI-iTAG detection](#PAI-iTAG-detection)
- [COCO2017](#COCO2017)
- [VOC2007](#VOC2007)
- [VOC2012](#VOC2012)

### PAI-iTAG detection

`PAI-iTAG` is a platform for intelligent data annotation, which supports the annotation of various data types such as images, texts, videos, and audios, as well as multi-modal mixed annotation.

Please refer to [智能标注iTAG](https://help.aliyun.com/document_detail/311162.html) for file format and data annotation.

#### Download

Download [SmallCOCO](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/unittest/data/detection/small_coco_itag/small_coco_itag.tar.gz) dataset to `data/demo_itag_coco`,
Directory structure should be as follows:

```text
data/demo_itag_coco/
├── train2017
├── train2017_20_local.manifest
├── val2017
└── val2017_20_local.manifest
```

### COCO2017

The COCO dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

The COCO dataset has been updated for several editions, and coco2017 is widely used. In 2017, the training/validation split was 118K/5K and test set is a subset of 41K images of the 2015 test set.

For more detailed information, please refer to [COCO](https://cocodataset.org/#home).

#### Download

Download  [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) (18G) ,[val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (1G), [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (241MB) and uncompress files to to `data/coco2017`.

Directory structure is as follows:

```text
data/coco2017
└── annotations
    ├── instances_train2017.json
    ├── instances_val2017.json
└── train2017
    ├── 000000000009.jpg
    ├── 000000000025.jpg
    ├── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── ...
```

### VOC2007

PASCAL VOC 2007 is a dataset for image recognition. The twenty object classes that have been selected are:

- *Person:* person
- *Animal:* bird, cat, cow, dog, horse, sheep
- *Vehicle:* aeroplane, bicycle, boat, bus, car, motorbike, train
- *Indoor:* bottle, chair, dining table, potted plant, sofa, tv/monitor

Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations.

For more detailed information, please refer to [voc2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html).

#### Download

Download [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (439MB) and uncompress files to to `data/VOCdevkit`.

Directory structure is as follows:

```
data/VOCdevkit
└── VOC2007
    └── Annotations
        ├── 000005.xml
        ├── 001010.xml
    	├── ...
    └── JPEGImages
        ├── 000005.jpg
        ├── 001010.jpg
        ├── ...
    └── SegmentationClass
        ├── 000005.png
        ├── 001010.png
        ├── ...
    └── SegmentationObject
        ├── 000005.png
        ├── 001010.png
        ├── ...
    └── ImageSets
        └── Layout
            ├── train.txt
            ├── trainval.txt
            ├── val.txt
        └── Main
            ├── train.txt
            ├── val.txt
            ├── ...
        └── Segmentation
            ├── train.txt
            ├── trainval.txt
            ├── val.txt
```

### VOC2012

The PASCAL VOC 2012 dataset contains 20 object categories including:

- *Person:* person
- *Animal:* bird, cat, cow, dog, horse, sheep
- *Vehicle:* aeroplane, bicycle, boat, bus, car, motorbike, train
- *Indoor:* bottle, chair, dining table, potted plant, sofa, tv/monitor

Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations.

For more detailed information, please refer to [voc2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

#### Download

Download [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2G) and uncompress files to to `data/VOCdevkit`.

Directory structure is as follows:

```
data/VOCdevkit
└── VOC2012
    └── Annotations
        ├── 000005.xml
        ├── 001010.xml
    	├── ...
    └── JPEGImages
        ├── 000005.jpg
        ├── 001010.jpg
        ├── ...
    └── SegmentationClass
        ├── 000005.png
        ├── 001010.png
        ├── ...
    └── SegmentationObject
        ├── 000005.png
        ├── 001010.png
        ├── ...
    └── ImageSets
        └── Layout
            ├── train.txt
            ├── trainval.txt
            ├── val.txt
        └── Main
            ├── train.txt
            ├── val.txt
            ├── ...
        └── Segmentation
            ├── train.txt
            ├── trainval.txt
            ├── val.txt
```

## Self-Supervised Learning

- [Imagenet-1k](#SSL-Imagenet-1k)
- [Imagenet-1k-TFrecords](#SSL-Imagenet-1k-TFrecords)

### Imagenet-1k<span id="SSL-Imagenet-1k"></span>

Refer to [Image Classification: Imagenet-1k](#Imagenet-1k).

### Imagenet-1k-TFrecords<span id="SSL-Imagenet-1k-TFrecords"></span>

Refer to [Image Classification: Imagenet-1k-TFrecords](#Imagenet-1k-TFrecords).

## Pose (Keypoint)

- [COCO2017](#Pose-COCO2017)

### COCO2017<span id="Pose-COCO2017"></span>

The COCO dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

The COCO dataset has been updated for several editions, and coco2017 is widely used. In 2017, the training/validation split was 118K/5K and test set is a subset of 41K images of the 2015 test set.

For more detailed information, please refer to [COCO](https://cocodataset.org/#home).

#### Download

Download it as follows:

1. Download data: [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) (18G) , [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (1G)
2. Download annotations: [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (241MB)
3. Download person detection results: [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) (26.2MB).

Then uncompress files to `data/coco2017`, directory structure is as follows:

```text
data/coco2017
└── annotations
    ├── person_keypoints_train2017.json
    ├── person_keypoints_val2017.json
└── person_detection_results
    ├── COCO_val2017_detections_AP_H_56_person.json
    ├── COCO_test-dev2017_detections_AP_H_609_person.json
└── train2017
    ├── 000000000009.jpg
    ├── 000000000025.jpg
    ├── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── ...
```
