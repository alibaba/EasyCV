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

COCO_STUFF_CLASSES = [
    'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
    'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush',
    'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush',
    'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other',
    'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other',
    'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill',
    'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss',
    'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow',
    'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
    'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea',
    'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
    'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other',
    'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'waterdrops', 'window-blind', 'window-other', 'wood'
]

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

BASE_OSS_PATH = 'oss://pai-vision-data-hz/unittest/'
BASE_LOCAL_PATH = os.path.expanduser('~/easycv_nfs/')

TMP_DIR_OSS = os.path.join(BASE_OSS_PATH, 'tmp')
TMP_DIR_LOCAL = os.path.join(BASE_LOCAL_PATH, 'tmp')

CLS_DATA_ITAG_OSS = os.path.join(
    BASE_OSS_PATH,
    'local_backup/easycv_nfs/data/classification/cls_itagtest/cls_itagtest.manifest'
)
CLS_TRAIN_TEST = os.path.join(
    BASE_LOCAL_PATH,
    'data/classification/class_list_test/cls_itagtest_imagelist.txt')
CLS_DATA_NPY_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/npy/')
SMALL_IMAGENET_RAW_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/small_imagenet_raw')
CIFAR10_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/cifar10')
CIFAR100_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/classification/cifar100')
SAMLL_IMAGENET1K_RAW_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                          'datasets/imagenet-1k/imagenet_raw')
CLASS_LIST_TEST = os.path.join(BASE_LOCAL_PATH,
                               'data/classification/class_list_test')

SMALL_IMAGENET_TFRECORD_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/small_imagenet_tfrecord/')
IMAGENET_LABEL_TXT = os.path.join(
    BASE_LOCAL_PATH, 'data/classification/imagenet/imagenet_label.txt')
CLS_DATA_NPY_OSS = os.path.join(BASE_OSS_PATH, 'data/classification/npy/')
SMALL_IMAGENET_TFRECORD_OSS = os.path.join(
    BASE_OSS_PATH, 'data/classification/small_imagenet_tfrecord/')
SMALL_MARKET1501 = os.path.join(BASE_LOCAL_PATH,
                                'data/tracking/small_Market1501')
TEST_MOT_DIR = os.path.join(BASE_LOCAL_PATH, 'data/tracking/mot20_1.mp4')

IO_DATA_TXTX_OSS = os.path.join(BASE_OSS_PATH, 'data/io_test_dir/txts/')
IO_DATA_MULTI_DIRS_OSS = os.path.join(BASE_OSS_PATH,
                                      'data/io_test_dir/multi_dirs/')
DET_DATA_SMALL_COCO_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                         'data/detection/small_coco')
CLS_DATA_COMMON_LOCAL = os.path.join(BASE_LOCAL_PATH, 'download_local/cls')
DET_DATASET_DOWNLOAD_SMALL = os.path.join(
    BASE_LOCAL_PATH, 'download_local/small_download/detection')
DET_DATA_COCO2017_DOWNLOAD = os.path.join(BASE_LOCAL_PATH, 'download_local/')
VOC_DATASET_DOWNLOAD_LOCAL = os.path.join(BASE_LOCAL_PATH, 'download_local')
VOC_DATASET_DOWNLOAD_SMALL = os.path.join(BASE_LOCAL_PATH,
                                          'download_local/small_download')
COCO_DATASET_DOWNLOAD_SMALL = os.path.join(BASE_LOCAL_PATH,
                                           'download_local/small_download')
CONFIG_PATH = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'

DET_DATA_RAW_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/detection/raw_data')
DET_DATA_SMALL_VOC_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                        'data/detection/small_voc')
DET_DATASET_DOWNLOAD_WIDER_PERSON_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'data/detection/small_widerPerson')
DET_DATASET_DOWNLOAD_AFRICAN_WILDLIFE = os.path.join(
    BASE_LOCAL_PATH, 'data/detection/small_african_wildlife')
DET_DATASET_FRUIT = os.path.join(BASE_LOCAL_PATH, 'data/detection/small_fruit')
DET_DATASET_PET = os.path.join(
    BASE_LOCAL_PATH, 'data/detection/small_pet/annotations/annotations')
DET_DATASET_ARTAXOR = os.path.join(BASE_LOCAL_PATH,
                                   'data/detection/small_artaxor')
DET_DATASET_TINY_PERSON = os.path.join(BASE_LOCAL_PATH,
                                       'data/detection/small_tiny_person')
DET_DATASET_WIDER_FACE = os.path.join(BASE_LOCAL_PATH,
                                      'data/detection/small_widerface')
DET_DATASET_CROWD_HUMAN = os.path.join(BASE_LOCAL_PATH,
                                       'data/detection/small_crowdhuman')
DET_DATASET_OBJECT365 = os.path.join(BASE_LOCAL_PATH,
                                     'data/detection/small_object365')

DET_DATA_MANIFEST_OSS = os.path.join(BASE_OSS_PATH,
                                     'data/detection/small_coco_itag')

POSE_DATA_SMALL_COCO_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                          'data/pose/small_coco')
POSE_DATA_CROWDPOSE_SMALL_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                               'data/pose/small_CrowdPose/')
POSE_DATA_OC_HUMAN_SMALL_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                              'data/pose/small_oc_human/')
POSE_DATA_MPII_DOWNLOAD_SMALL_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'download_local/small_download/pose/small_mpii/')

SSL_SMALL_IMAGENET_FEATURE = os.path.join(
    BASE_LOCAL_PATH, 'data/selfsup/small_imagenet_feature')
SSL_SMALL_IMAGENET_RAW = os.path.join(BASE_LOCAL_PATH,
                                      'data/selfsup/small_imagenet')
TEST_IMAGES_DIR = os.path.join(BASE_LOCAL_PATH, 'data/test_images')

COMPRESSION_TEST_DATA = os.path.join(BASE_LOCAL_PATH,
                                     'data/compression/test_data')
# Seg data
SEG_DATA_SMALL_RAW_LOCAL = os.path.join(BASE_LOCAL_PATH,
                                        'data/segmentation/small_voc_200')
SEG_DATA_SMALL_VOC_DOWNLOAD_LOCAL = os.path.join(
    BASE_LOCAL_PATH, 'download_local/small_download/segmentation')
SEG_DATA_SMALL_COCO_STUFF_10K = os.path.join(
    BASE_LOCAL_PATH, 'data/segmentation/small_coco_stuff/small_coco_stuff10k')
SEG_DATA_SAMLL_COCO_STUFF_164K = os.path.join(
    BASE_LOCAL_PATH, 'data/segmentation/small_coco_stuff/small_coco_stuff164k')
SEG_DATA_SAMLL_CITYSCAPES = os.path.join(BASE_LOCAL_PATH,
                                         'data/segmentation/small_cityscapes')

# OCR data
SMALL_OCR_CLS_DATA = os.path.join(BASE_LOCAL_PATH, 'data/ocr/small_ocr_cls')
SMALL_OCR_DET_DATA = os.path.join(BASE_LOCAL_PATH, 'data/ocr/small_ocr_det')
SMALL_OCR_DET_PAI_DATA = os.path.join(BASE_LOCAL_PATH,
                                      'data/ocr/small_ocr_det_pai')
SMALL_OCR_REC_DATA = os.path.join(BASE_LOCAL_PATH, 'data/ocr/small_ocr_rec')

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
    BASE_LOCAL_PATH, 'pretrained_models/detection/infer_yolox/epoch_300.pt')
PRETRAINED_MODEL_YOLOXS_EXPORT_OLD = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/detection/infer_yolox/old.pt')
PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_nopre_notrt_e2e.pt.jit')
PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_pre_notrt_e2e.pt.jit')
PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT_B2 = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_pre_notrt_e2e_b2.pt.jit'
)
PRETRAINED_MODEL_YOLOXS_NOPRE_TRT_JIT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_nopre_trt.pt.jit')
PRETRAINED_MODEL_YOLOXS_PRE_TRT_JIT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_pre_trt.pt.jit')
PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_BLADE = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_nopre_notrt.pt.blade')
PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_BLADE = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_pre_notrt.pt.blade')
PRETRAINED_MODEL_YOLOXS_NOPRE_TRT_BLADE = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_nopre_trt.pt.blade')
PRETRAINED_MODEL_YOLOXS_PRE_TRT_BLADE = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection/infer_yolox/epoch_300_pre_trt.pt.blade')
PRETRAINED_MODEL_YOLOXS = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/detection/infer_yolox/epoch_300.pth')

PRETRAINED_MODEL_POSE_HRNET_EXPORT = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/pose/hrnet/pose_hrnet_epoch_210_export.pt')
PRETRAINED_MODEL_YOLOX_COMPRESSION = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/compression/yolox_compression.pth')
PRETRAINED_MODEL_MAE = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/classification/vit/mae_vit_b_1600.pth')
PRETRAINED_MODEL_MASK2FORMER_DIR = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/segmentation/mask2former/')
PRETRAINED_MODEL_MASK2FORMER = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                            'mask2former_r50_instance.pth')
PRETRAINED_MODEL_OCRDET = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/ocr/det/student_export.pth')
PRETRAINED_MODEL_OCRREC = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/ocr/rec/best_accuracy_student_export.pth')
PRETRAINED_MODEL_OCRCLS = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/ocr/cls/best_accuracy_export.pth')
PRETRAINED_MODEL_SEGFORMER = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/segmentation/segformer/segformer_b0/SegmentationEvaluator_mIoU_best.pth'
)
PRETRAINED_MODEL_BEVFORMER_BASE = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/detection3d/bevformer/bevformer_base_epoch_24.pth')
PRETRAINED_MODEL_FACE_2D_KEYPOINTS = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/face_2d_keypoints/epoch_400.pth')
PRETRAINED_MODEL_HAND_KEYPOINTS = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/pose/hand/hrnet/hrnet_w18_256x256.pth')
PRETRAINED_MODEL_WHOLEBODY_DETECTION = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/pose/wholebody/epoch_290.pth')
PRETRAINED_MODEL_WHOLEBODY = os.path.join(
    BASE_LOCAL_PATH,
    'pretrained_models/pose/wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
)
PRETRAINED_MODEL_X3D_XS = os.path.join(
    BASE_LOCAL_PATH, 'pretrained_models/video/x3d/epoch_300.pth')
MODEL_CONFIG_SEGFORMER = (
    './configs/segmentation/segformer/segformer_b0_coco.py')
SMALL_COCO_WHOLE_BODY_HAND_ROOT = os.path.join(
    BASE_LOCAL_PATH, 'data/pose/hand/small_whole_body_hand_coco')
SMALL_NUSCENES_PATH = os.path.join(
    BASE_LOCAL_PATH, 'data/detection3d/nuScenes/nuscenes-v1.0-mini')
SMALL_COCO_WHOLEBODY_ROOT = os.path.join(BASE_LOCAL_PATH,
                                         'data/pose/wholebody/data')
MODEL_CONFIG_MASK2FORMER_PAN = (
    './configs/segmentation/mask2former/mask2former_r50_8xb2_e50_panoptic.py')
MODEL_CONFIG_MASK2FORMER_INS = (
    './configs/segmentation/mask2former/mask2former_r50_8xb2_e50_instance.py')
MODEL_CONFIG_MASK2FORMER_SEM = (
    './configs/segmentation/mask2former/mask2former_r50_8xb2_e127_semantic.py')
VIDEO_DATA_SMALL_RAW_LOCAL = os.path.join(BASE_LOCAL_PATH, 'data/video')
