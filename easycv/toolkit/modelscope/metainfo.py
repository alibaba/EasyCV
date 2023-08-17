# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import CustomDatasets, Models, Pipelines


class EasyCVModels(Models):
    yolox = 'YOLOX'
    segformer = 'Segformer'
    hand_2d_keypoints = 'HRNet-Hand2D-Keypoints'
    image_object_detection_auto = 'image-object-detection-auto'
    dino = 'DINO'


class EasyCVPipelines(Pipelines):
    easycv_detection = 'easycv-detection'
    easycv_segmentation = 'easycv-segmentation'
    image_panoptic_segmentation_easycv = 'image-panoptic-segmentation-easycv'


class EasyCVCustomDatasets(CustomDatasets):
    """ Names for different datasets.
    """
    ClsDataset = 'ClsDataset'
    Face2dKeypointsDataset = 'FaceKeypointDataset'
    HandCocoWholeBodyDataset = 'HandCocoWholeBodyDataset'
    HumanWholeBodyKeypointDataset = 'WholeBodyCocoTopDownDataset'
    SegDataset = 'SegDataset'
    DetDataset = 'DetDataset'
    DetImagesMixDataset = 'DetImagesMixDataset'
