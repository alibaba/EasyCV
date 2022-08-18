# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
import logging
import math

import torch
import torch.nn as nn

from easycv.models.backbones.darknet import CSPDarknet
from easycv.models.backbones.network_blocks import (BaseConv, CSPLayer, DWConv,
                                                    GSConv, VoVGSCSP)
from easycv.models.backbones.repvgg_yolox_backbone import RepVGGYOLOX
from easycv.models.registry import BACKBONES
from .asff import ASFF


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    param_map = {
        'nano': [0.33, 0.25],
        'tiny': [0.33, 0.375],
        's': [0.33, 0.5],
        'm': [0.67, 0.75],
        'l': [1.0, 1.0],
        'x': [1.33, 1.25]
    }

    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 backbone='CSPDarknet',
                 neck_type='yolo',
                 neck_mode='all',
                 in_features=('dark3', 'dark4', 'dark5'),
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 act='silu',
                 use_att=None,
                 asff_channel=2,
                 expand_kernel=3):
        super().__init__()

        # build backbone
        if backbone == 'CSPDarknet':
            self.backbone = CSPDarknet(
                depth, width, depthwise=depthwise, act=act)
        elif backbone == 'RepVGGYOLOX':
            self.backbone = RepVGGYOLOX(
                in_channels=3, depth=depth, width=width)
        else:
            logging.warning(
                'YOLOX-PAI backbone must in [CSPDarknet, RepVGGYOLOX], otherwise we use RepVGGYOLOX as default'
            )
            self.backbone = RepVGGYOLOX(
                in_channels=3, depth=depth, width=width)

        self.backbone_name = backbone

        # build neck
        self.in_features = in_features
        self.in_channels = in_channels

        Conv = DWConv if depthwise else BaseConv
        self.neck_type = neck_type
        self.neck_mode = neck_mode
        if neck_type != 'gsconv':
            if neck_type != 'yolo':
                logging.warning(
                    'YOLOX-PAI backbone must in [yolo, gsconv], otherwise we use yolo as default'
                )
            self.neck_type = 'yolo'

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.lateral_conv0 = BaseConv(
                int(in_channels[2] * width),
                int(in_channels[1] * width),
                1,
                1,
                act=act)
            self.C3_p4 = CSPLayer(
                int(2 * in_channels[1] * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)  # cat

            self.reduce_conv1 = BaseConv(
                int(in_channels[1] * width),
                int(in_channels[0] * width),
                1,
                1,
                act=act)
            self.C3_p3 = CSPLayer(
                int(2 * in_channels[0] * width),
                int(in_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)

            # bottom-up conv
            self.bu_conv2 = Conv(
                int(in_channels[0] * width),
                int(in_channels[0] * width),
                3,
                2,
                act=act)
            self.C3_n3 = CSPLayer(
                int(2 * in_channels[0] * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)

            # bottom-up conv
            self.bu_conv1 = Conv(
                int(in_channels[1] * width),
                int(in_channels[1] * width),
                3,
                2,
                act=act)
            self.C3_n4 = CSPLayer(
                int(2 * in_channels[1] * width),
                int(in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.gsconv1 = GSConv(
                int(in_channels[2] * width),
                int(in_channels[1] * width),
                1,
                1,
                act=act)

            self.gsconv2 = GSConv(
                int(in_channels[1] * width),
                int(in_channels[0] * width),
                1,
                1,
                act=act)

            self.gsconv4 = GSConv(
                int(in_channels[0] * width),
                int(in_channels[0] * width),
                3,
                2,
                act=act)

            self.gsconv5 = GSConv(
                int(in_channels[1] * width),
                int(in_channels[1] * width),
                3,
                2,
                act=act)

            if self.neck_mode == 'all':
                self.vovGSCSP1 = VoVGSCSP(
                    int(2 * in_channels[1] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                )

                self.gsconv3 = GSConv(
                    int(2 * in_channels[0] * width),
                    int(2 * in_channels[0] * width),
                    1,
                    1,
                    act=act)
                self.vovGSCSP2 = VoVGSCSP(
                    int(2 * in_channels[0] * width),
                    int(in_channels[0] * width),
                    round(3 * depth),
                    False,
                )

                self.vovGSCSP3 = VoVGSCSP(
                    int(2 * in_channels[0] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                )

                self.vovGSCSP4 = VoVGSCSP(
                    int(2 * in_channels[1] * width),
                    int(in_channels[2] * width),
                    round(3 * depth),
                    False,
                )
            else:
                self.C3_p4 = CSPLayer(
                    int(2 * in_channels[1] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act)  # cat

                self.C3_p3 = CSPLayer(
                    int(2 * in_channels[0] * width),
                    int(in_channels[0] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act)

                self.C3_n3 = CSPLayer(
                    int(2 * in_channels[0] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act)

                self.C3_n4 = CSPLayer(
                    int(2 * in_channels[1] * width),
                    int(in_channels[2] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act)

        # build attention after PAN
        self.use_att = use_att
        default_attention_list = ['ASFF', 'ASFF_sim']
        if use_att is not None and use_att not in default_attention_list:
            logging.warning(
                'YOLOX-PAI backbone must in [ASFF, ASFF_sim], otherwise we use ASFF as default'
            )

        if self.use_att == 'ASFF' or self.use_att == 'ASFF_sim':
            self.asff_1 = ASFF(
                level=0,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act,
            )
            self.asff_2 = ASFF(
                level=1,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act,
            )
            self.asff_3 = ASFF(
                level=2,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act,
            )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        if self.backbone_name == 'CSPDarknet':
            out_features = self.backbone(input)
            features = [out_features[f] for f in self.in_features]
            [x2, x1, x0] = features
        else:
            features = self.backbone(input)
            [x2, x1, x0] = features

        if self.neck_type == 'yolo':
            fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            f_out0 = self.C3_p4(f_out0)  # 1024->512/16

            fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
            pan_out2 = self.C3_p3(f_out1)  # 512->256/8

            p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
            pan_out1 = self.C3_n3(p_out1)  # 512->512/16

            p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
            pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        else:
            # gsconv
            fpn_out0 = self.gsconv1(x0)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            if self.neck_mode == 'all':
                f_out0 = self.vovGSCSP1(f_out0)  # 1024->512/16
            else:
                f_out0 = self.C3_p4(f_out0)

            fpn_out1 = self.gsconv2(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8

            if self.neck_mode == 'all':
                f_out1 = self.gsconv3(f_out1)
                pan_out2 = self.vovGSCSP2(f_out1)  # 512->256/8
            else:
                pan_out2 = self.C3_p3(f_out1)  # 512->256/8

            p_out1 = self.gsconv4(pan_out2)  # 256->256/16
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
            if self.neck_mode == 'all':
                pan_out1 = self.vovGSCSP3(p_out1)  # 512->512/16
            else:
                pan_out1 = self.C3_n3(p_out1)  # 512->512/16

            p_out0 = self.gsconv5(pan_out1)  # 512->512/32
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
            if self.neck_mode == 'all':
                pan_out0 = self.vovGSCSP4(p_out0)  # 1024->1024/32
            else:
                pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        # forward for attention
        if self.use_att == 'ASFF' or self.use_att == 'ASFF_sim':
            pan_out0 = self.asff_1(outputs)
            pan_out1 = self.asff_2(outputs)
            pan_out2 = self.asff_3(outputs)
            outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
