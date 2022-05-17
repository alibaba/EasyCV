# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.

import torch
import torch.nn as nn

from easycv.models.backbones.darknet import CSPDarknet
from easycv.models.backbones.network_blocks import BaseConv, CSPLayer, DWConv
from .attention import SE, CBAM, ECA
from .ASFF import ASFF

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=('dark3', 'dark4', 'dark5'),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act='silu',
        use_att=None
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

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

        self.use_att=use_att

        if self.use_att!=None and self.use_att!='ASFF':
            # add attention layer
            if self.use_att=="CBAM":
                ATT = CBAM
            elif self.use_att=="SE":
                ATT = SE
            elif self.use_att=="ECA":
                ATT = ECA
            else:
                assert "Unknown Attention Layer!"

            self.att_1 = ATT(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道
            self.att_2 = ATT(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
            self.att_3 = ATT(int(in_channels[0] * width))  # 对应dark3输出的256维度通道

        if self.use_att=='ASFF':
            self.asff_1 = ASFF(level=0, multiplier=width)
            self.asff_2 = ASFF(level=1, multiplier=width)
            self.asff_3 = ASFF(level=2, multiplier=width)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # add attention
        if self.use_att!=None and self.use_att!='ASFF':
            x0 = self.att_1(x0)
            x1 = self.att_2(x1)
            x2 = self.att_3(x2)

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

        outputs = (pan_out2, pan_out1, pan_out0)

        if self.use_att=='ASFF':
            pan_out0 = self.asff_1(outputs)
            pan_out1 = self.asff_2(outputs)
            pan_out2 = self.asff_3(outputs)
            outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
