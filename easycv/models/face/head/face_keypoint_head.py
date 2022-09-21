import copy

import numpy as np
import torch
import torch.nn as nn

from easycv.models.builder import HEADS, build_loss
from easycv.models.utils.face_keypoint_utils import (InvertedResidual, View,
                                                     conv_bn, conv_no_relu,
                                                     get_keypoint_accuracy)


@HEADS.register_module
class FaceKeypointHead(nn.Module):

    def __init__(
        self,
        mean_face,
        loss_keypoint,
        in_channels=48,
        out_channels=212,
        input_size=96,
        inverted_expand_ratio=2,
        inverted_activation='half_v2',
    ):
        super(FaceKeypointHead, self).__init__()
        self.input_size = input_size
        self.face_mean_shape = copy.deepcopy(np.asarray(mean_face))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.branches = []

        self.loss = build_loss(loss_keypoint)

        # points
        self.branches.append(
            nn.Sequential(
                InvertedResidual(
                    in_channels,
                    96,
                    3,
                    1,
                    1,
                    expand_ratio=inverted_expand_ratio,
                    activation=inverted_activation),
                View((-1, 96 * 3 * 3, 1, 1)), conv_bn(96 * 3 * 3, 128, 1, 1,
                                                      0),
                conv_bn(128, 128, 1, 1, 0),
                conv_no_relu(128, out_channels, 1, 1, 0),
                View((-1, out_channels))))
        self.branches = nn.ModuleList(self.branches)

    def get_loss(self, output, target_point, target_point_mask, target_pose):
        losses = dict()
        loss = self.loss(output * target_point_mask, target_point, target_pose)
        losses['point_loss'] = loss

        return losses

    def get_accuracy(self, output, target_point):
        return get_keypoint_accuracy(output, target_point)

    def forward(self, x):
        point = self.branches[0](x)
        point = point * 0.5 + torch.from_numpy(self.face_mean_shape).to(
            self.device)
        point = point * self.input_size

        return point
