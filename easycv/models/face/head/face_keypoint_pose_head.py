import numpy as np
import torch
import torch.nn as nn

from easycv.models.builder import HEADS, build_loss
from easycv.models.utils.face_keypoint_utils import (InvertedResidual, View,
                                                     conv_bn, conv_no_relu,
                                                     get_pose_accuracy)


@HEADS.register_module
class FacePoseHead(nn.Module):

    def __init__(
        self,
        loss_pose,
        in_channels=48,
        out_channels=3,
        inverted_expand_ratio=2,
        inverted_activation='half_v2',
    ):
        super(FacePoseHead, self).__init__()
        self.branches = []

        self.loss = build_loss(loss_pose)

        # pose
        self.branches.append(
            nn.Sequential(
                InvertedResidual(
                    in_channels,
                    48,
                    3,
                    1,
                    1,
                    expand_ratio=inverted_expand_ratio,
                    activation=inverted_activation),
                View((-1, 48 * 3 * 3, 1, 1)), conv_bn(48 * 3 * 3, 48, 1, 1, 0),
                conv_bn(48, 48, 1, 1, 0),
                conv_no_relu(48, out_channels, 1, 1, 0),
                View((-1, out_channels))))
        self.branches = nn.ModuleList(self.branches)

    def get_loss(self, output, target_pose):
        losses = dict()
        loss = self.loss(output, target_pose)
        losses['pose_loss'] = loss

        return losses

    def get_accuracy(self, output, target_pose):
        return get_pose_accuracy(output, target_pose)

    def forward(self, x):
        return self.branches[0](x)
