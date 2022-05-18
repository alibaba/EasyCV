import os

IMG_NORM_CFG_255 = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
IMG_NORM_CFG = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

BASE_OSS_PATH = 'oss://pai-vision-data-hz/unittest/'
BASE_LOCAL_PATH = os.path.expanduser('~/easycv_nfs/')

TMP_DIR_OSS = os.path.join(BASE_OSS_PATH, 'tmp')
TMP_DIR_LOCAL = os.path.join(BASE_LOCAL_PATH, 'tmp')

CLS_DATA_NPY_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/npy/')
SMALL_IMAGENET_RAW_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/small_imagenet_raw')
CIFAR10_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/cifar10')
CIFAR100_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/cifar100')
SMALL_IMAGENET_TFRECORD_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/small_imagenet_tfrecord/')
IMAGENET_LABEL_TXT = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/imagenet/imagenet_label.txt')
CLS_DATA_NPY_OSS = os.path.join(BASE_OSS_PATH, 'data/classification/npy/')
SMALL_IMAGENET_TFRECORD_OSS = os.path.join(
    BASE_OSS_PATH, 'data/classification/small_imagenet_tfrecord/')

IO_DATA_TXTX_OSS = os.path.join(BASE_OSS_PATH, 'data/io_test_dir/txts/')

DET_DATA_SMALL_COCO_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                         'data/detection/small_coco')
DET_DATA_RAW_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/detection/raw_data')
DET_DATA_SMALL_VOC_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                        'data/detection/small_voc')
DET_DATA_MANIFEST_OSS = os.path.join(BASE_OSS_PATH,
                                     'data/detection/small_coco_itag')

POSE_DATA_SMALL_COCO_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                          'data/pose/small_coco')

SSL_SMALL_IMAGENET_FEATURE = os.path.join(
    BASE_LOCAL_PATH, 'data/selfsup/small_imagenet_feature')
SSL_SMALL_IMAGENET_RAW = os.path.join(BASE_LOCAL_PATH,
                                      'data/selfsup/small_imagenet')
TEST_IMAGES_DIR = os.path.join(BASE_LOCAL_PATH, 'data/test_images')

COMPRESSION_TEST_DATA = os.path.join(BASE_LOCAL_PATH,
                                     'data/compression/test_data')

PRETRAINED_MODEL_MOCO = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/selfsup/moco/moco_epoch_200.pth')
PRETRAINED_MODEL_RESNET50 = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/classification/resnet/resnet50.pth')
PRETRAINED_MODEL_RESNET50_WITHOUTHEAD = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/classification/resnet/resnet50_withhead.pth')
PRETRAINED_MODEL_FACEID = os.path.join(BASE_LOCAL_PATH,
                                       'pretrained_models/faceid')
PRETRAINED_MODEL_YOLOXS_EXPORT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/yolox_s/yolox_s_epoch_300_export.pt')
PRETRAINED_MODEL_YOLOXS = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/yolox_s/yolox_s_epoch_300.pth')
PRETRAINED_MODEL_POSE_HRNET_EXPORT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/pose/hrnet/pose_hrnet_epoch_210_export.pt')
PRETRAINED_MODEL_YOLOX_COMPRESSION = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/compression/yolox_compression.pth')
PRETRAINED_MODEL_MAE = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/classification/vit/mae_vit_b_1600.pth')
