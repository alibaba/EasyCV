import copy
import math

import numpy as np
import torch
import torch.nn as nn


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.PReLU(oup))


def conv_no_relu(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup))


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Softmax(nn.Module):

    def __init__(self, dim):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim)

    def forward(self, x):
        return self.softmax(x)


class InvertedResidual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio=2,
                 use_connect=False,
                 activation='relu'):
        super(InvertedResidual, self).__init__()

        hid_channels = int(inp * expand_ratio)
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        elif activation == 'prelu':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.PReLU(hid_channels),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.PReLU(hid_channels),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        elif activation == 'half_v1':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.PReLU(hid_channels),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        elif activation == 'half_v2':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.PReLU(hid_channels),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        self.use_connect = use_connect

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Residual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 kernel_size,
                 stride,
                 padding,
                 use_connect=False,
                 activation='relu'):
        super(Residual, self).__init__()

        self.use_connect = use_connect

        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        elif activation == 'prelu':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.PReLU(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.PReLU(oup))
        elif activation == 'half_v1':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.PReLU(oup))
        elif activation == 'half_v2':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.PReLU(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def pose_accuracy(output, target):
    with torch.no_grad():
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        acc = np.mean(np.abs(output - target))
        return acc


def ION(output, target, left_eye_left_coner_idx, right_eye_right_corner_idx,
        num_pts):
    with torch.no_grad():
        output = output.view(-1, num_pts, 2).cpu().numpy()
        target = target.view(-1, num_pts, 2).cpu().numpy()

        interocular = target[:,
                             left_eye_left_coner_idx] - target[:,
                                                               right_eye_right_corner_idx]
        interocular = np.sqrt(
            np.square(interocular[:, 0]) + np.square(interocular[:, 1])) + 1e-5
        dist = target - output
        dist = np.sqrt(np.square(dist[:, :, 0]) + np.square(dist[:, :, 1]))
        dist = np.sum(dist, axis=1)
        nme = dist / (interocular * num_pts)

    return np.mean(nme)


def get_keypoint_accuracy(output, target_point):
    accuracy = dict()
    num_points = 106
    left_eye_left_corner_index = 66
    right_eye_right_corner_index = 79

    nme = ION(output, target_point, left_eye_left_corner_index,
              right_eye_right_corner_index, num_points)

    accuracy['nme'] = nme

    return accuracy


def get_pose_accuracy(output, target_pose):
    accuracy = dict()
    pose_acc = pose_accuracy(output, target_pose)
    accuracy['pose_acc'] = float(pose_acc)
    return accuracy
