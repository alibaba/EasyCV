import copy
import math

import numpy as np
import torch
import torch.nn as nn

from easycv.models.builder import LOSSES

CONSTANT_CONTOUR = 66
CONSTANT_EYEBROW = 18
CONSTANT_EYE = 18
CONSTANT_NOSE = 30
CONSTANT_LIPS = 40
CONSTANT_EYE_CENTER = 4


@LOSSES.register_module()
class WingLossWithPose(nn.Module):

    def __init__(self,
                 num_points=106,
                 left_eye_left_corner_index=66,
                 right_eye_right_corner_index=79,
                 points_weight=1.0,
                 contour_weight=1.5,
                 eyebrow_weight=1.5,
                 eye_weight=1.7,
                 nose_weight=1.3,
                 lip_weight=1.7,
                 omega=10,
                 epsilon=2):
        super(WingLossWithPose, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

        self.num_points = num_points
        self.left_eye_left_corner_index = left_eye_left_corner_index
        self.right_eye_right_corner_index = right_eye_right_corner_index
        self.points_weight = points_weight
        contour_weight = np.full(CONSTANT_CONTOUR, contour_weight)
        eyebrow_left_weight = np.full(CONSTANT_EYEBROW, eyebrow_weight)
        eyebrow_right_weight = np.full(CONSTANT_EYEBROW, eyebrow_weight)
        nose_weight = np.full(CONSTANT_NOSE, nose_weight)
        eye_left_weight = np.full(CONSTANT_EYE, eye_weight)
        eye_right_weight = np.full(CONSTANT_EYE, eye_weight)
        lips_weight = np.full(CONSTANT_LIPS, lip_weight)
        eye_center_weight = np.full(CONSTANT_EYE_CENTER, eye_weight)
        part_weight = np.concatenate(
            (contour_weight, eyebrow_left_weight, eyebrow_right_weight,
             nose_weight, eye_left_weight, eye_right_weight, lips_weight,
             eye_center_weight),
            axis=0)

        self.part_weight = None
        if part_weight is not None:
            self.part_weight = torch.from_numpy(part_weight)

    def forward(self, pred, target, pose):
        weight = 5.0 * (1.0 - torch.cos(pose * np.pi / 180.0)) + 1.0
        weight = torch.sum(weight, dim=1) / 3.0
        weight = weight.view((weight.shape[0], 1))
        self.part_weight = self.part_weight.to(weight.device)
        if self.part_weight is not None:
            weight = weight * self.part_weight

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs() * weight
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss = delta_y2 - C
        result = self.points_weight * (loss1.sum() + loss.sum()) / (
            len(loss1) + len(loss))

        return result


@LOSSES.register_module()
class FacePoseLoss(nn.Module):

    def __init__(self, pose_weight=1.0):
        super(FacePoseLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.pose_weight = pose_weight

    def forward(self, pred, target):
        result = self.pose_weight * self.criterion(pred, target)
        return result
