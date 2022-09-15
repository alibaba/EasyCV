import mmcv
import numpy as np

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.models.utils.face_keypoint_utils import (get_keypoint_accuracy,
                                                     get_pose_accuracy)


@MODELS.register_module()
class FaceKeypoint(BaseModel):

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 pose_head=None,
                 pretrained=None,
                 loss_keypoint=None,
                 loss_pose=None):
        super().__init__()
        self.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            if 'loss_keypoint' not in keypoint_head and loss_keypoint is not None:
                keypoint_head['loss_keypoint'] = loss_keypoint
            self.keypoint_head = builder.build_head(keypoint_head)

        if pose_head is not None:
            if 'loss_pose' not in pose_head and loss_pose is not None:
                pose_head['loss_pose'] = loss_pose
            self.pose_head = builder.build_head(pose_head)

    @property
    def with_neck(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    @property
    def with_pose(self):
        """Check if has pose_head."""
        return hasattr(self, 'pose_head')

    def forward_train(self, img, target_point, target_point_mask, target_pose,
                      target_pose_mask, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)

        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output_points = self.keypoint_head(output)
        if self.with_pose:
            output_pose = self.pose_head(output)

        target_point = target_point * target_point_mask
        target_pose = target_pose * target_pose_mask

        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output_points, target_point, target_point_mask, target_pose)
            losses.update(keypoint_losses)
            keypoint_accuracy = get_keypoint_accuracy(output_points,
                                                      target_point)
            losses.update(keypoint_accuracy)

        if self.with_pose:
            output_pose = output_pose * 180.0 / np.pi
            output_pose = output_pose * target_pose_mask

            pose_losses = self.pose_head.get_loss(output_pose, target_pose)
            losses.update(pose_losses)
            pose_accuracy = get_pose_accuracy(output_pose, target_pose)
            losses.update(pose_accuracy)
        return losses

    def forward_test(self, img, **kwargs):
        """Defines the computation performed at every call when testing."""

        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output_points = self.keypoint_head(output)
        if self.with_pose:
            output_pose = self.pose_head(output)

        ret = {}
        ret['point'] = output_points
        ret['pose'] = output_pose
        return ret
