from .detection_pipeline import EasyCVDetectionPipeline
from .face_2d_keypoints_pipeline import Face2DKeypointsPipeline
from .hand_2d_keypoints_pipeline import Hand2DKeypointsPipeline
from .human_wholebody_keypoint_pipeline import HumanWholebodyKeypointsPipeline
from .image_panoptic_segmentation_pipeline import \
    ImagePanopticSegmentationEasyCVPipeline
from .segmentation_pipeline import EasyCVSegmentationPipeline

__all__ = [
    'EasyCVDetectionPipeline', 'EasyCVSegmentationPipeline',
    'Face2DKeypointsPipeline', 'HumanWholebodyKeypointsPipeline',
    'Hand2DKeypointsPipeline', 'ImagePanopticSegmentationEasyCVPipeline'
]
