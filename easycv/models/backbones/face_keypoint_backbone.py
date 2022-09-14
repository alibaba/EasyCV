import torch.nn as nn

from easycv.models.registry import BACKBONES
from easycv.models.utils.face_keypoint_utils import InvertedResidual, Residual


@BACKBONES.register_module
class FaceKeypointBackbone(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=48,
                 residual_activation='relu',
                 inverted_activation='half_v2',
                 inverted_expand_ratio=2):
        super(FaceKeypointBackbone, self).__init__()
        self.conv1 = Residual(in_channels, 12, 3, 2, 0)
        self.conv2 = Residual(12, 12, 3, 1, 0, activation=residual_activation)
        self.conv3 = Residual(12, 12, 3, 1, 1, activation=residual_activation)
        self.conv4 = Residual(12, 12, 3, 1, 0, activation=residual_activation)
        self.conv5 = Residual(12, 24, 3, 2, 0, activation=residual_activation)
        self.conv6 = Residual(24, 24, 3, 1, 0, activation=residual_activation)
        self.conv7 = Residual(24, 24, 3, 1, 1, activation=residual_activation)
        self.conv8 = Residual(24, 24, 3, 1, 1, activation=residual_activation)
        self.conv9 = InvertedResidual(
            24,
            48,
            3,
            2,
            0,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)
        self.conv10 = InvertedResidual(
            48,
            48,
            3,
            1,
            0,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)
        self.conv11 = InvertedResidual(
            48,
            48,
            3,
            1,
            1,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)
        self.conv12 = InvertedResidual(
            48,
            48,
            3,
            1,
            1,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)
        self.conv13 = InvertedResidual(
            48,
            48,
            3,
            1,
            1,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)
        self.conv14 = InvertedResidual(
            48,
            out_channels,
            3,
            2,
            0,
            expand_ratio=inverted_expand_ratio,
            activation=inverted_activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        return x14
