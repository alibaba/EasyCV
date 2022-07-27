# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.

import torch
import torch.nn as nn

from easycv.models.backbones.darknet import CSPDarknet
from easycv.models.backbones.efficientrep import EfficientRep
from easycv.models.backbones.network_blocks import BaseConv, CSPLayer, DWConv, GSConv, VoVGSCSP
from torchsummaryX import summary
import math


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


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
        asff_channel = 16,
        use_att=None,
        expand_kernel=3,
        down_rate=32,
        use_dconv=False,
        use_expand=True,
        spp_type='spp',
        backbone = "CSPDarknet",
        neck = 'gsconv',
        neck_mode = 'part',
    ):
        super().__init__()
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act,spp_type=spp_type)
        self.backbone_name = backbone
        if backbone == "CSPDarknet":
            self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        else:
            depth_mul = depth
            width_mul = width
            num_repeat_backbone = [1, 6, 12, 18, 6]
            channels_list_backbone = [64, 128, 256, 512, 1024]
            num_repeat_neck = [12, 12, 12, 12]
            channels_list_neck = [256, 128, 128, 256, 256, 512]

            channels = 3

            num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in
                          (num_repeat_backbone + num_repeat_neck)]

            channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
            self.backbone = EfficientRep(in_channels=channels, channels_list=channels_list, num_repeats=num_repeat)


        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.neck = neck
        self.neck_mode = neck_mode

        if neck =='yolo':
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
            # gsconv
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
                act=act
            )

            self.gsconv4 = GSConv(
                int(in_channels[0] * width),
                int(in_channels[0] * width),
                3,
                2,
                act=act
            )

            self.gsconv5 = GSConv(
                int(in_channels[1] * width),
                int(in_channels[1] * width),
                3,
                2,
                act=act
            )

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
                    act=act
                )
                self.vovGSCSP2 = VoVGSCSP(
                    int(2*in_channels[0] * width),
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



    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        # out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        #  backbone
        x2,x1,x0 = x
        if self.neck =='yolo':
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
            print('in')
            # gsconv
            fpn_out0 = self.gsconv1(x0)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            if self.neck_mode =='all':
                f_out0 = self.vovGSCSP1(f_out0)  # 1024->512/16
            else:
                f_out0 = self.C3_p4(f_out0)

            fpn_out1 = self.gsconv2(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
            if self.neck_mode =='all':
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

        return outputs

if __name__=='__main__':
    x = (torch.randn(1,128,80,80).cuda(),torch.randn(1,256,40,40).cuda(),torch.randn(1,512,20,20).cuda())
    model = YOLOPAFPN(depth=0.33, width=0.5).cuda()
    summary(model,x)