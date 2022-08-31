# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
import torch
import torch.nn as nn

from easycv.models.builder import HEADS
from .yolo_head_template import YOLOXHead_Template


@HEADS.register_module
class YOLOXHead(YOLOXHead_Template):

    def __init__(
        self,
        num_classes=80,
        model_type='s',
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act='silu',
        conv_type='conv',
        stage='CLOUD',
        obj_loss_type='BCE',
        reg_loss_type='giou',
        decode_in_inference=True,
        width=None,
    ):
        """
        Args:
            num_classes (int): detection class numbers.
            width (float): model width. Default value: 1.0.
            strides (list): expanded strides. Default value: [8, 16, 32].
            in_channels (list): model conv channels set. Default value: [256, 512, 1024].
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Default value: False.
            stage (str): model stage, distinguish edge head to cloud head. Default value: CLOUD.
            obj_loss_type (str): the loss function of the obj conf. Default value: l1.
            reg_loss_type (str): the loss function of the box prediction. Default value: l1.
        """
        super(YOLOXHead, self).__init__(
            width=width,
            num_classes=num_classes,
            model_type=model_type,
            strides=strides,
            in_channels=in_channels,
            act=act,
            conv_type=conv_type,
            stage=stage,
            obj_loss_type=obj_loss_type,
            reg_loss_type=reg_loss_type,
            decode_in_inference=decode_in_inference,
        )

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(
                        1, grid.shape[1]).fill_(stride_this_level).type_as(
                            xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4,
                                                 hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                if self.stage == 'EDGE':
                    m = nn.Hardsigmoid()
                    output = torch.cat(
                        [reg_output, m(obj_output),
                         m(cls_output)], 1)
                else:
                    output = torch.cat([
                        reg_output,
                        obj_output.sigmoid(),
                        cls_output.sigmoid()
                    ], 1)

            outputs.append(output)

        if self.training:

            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )

        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs],
                                dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs
