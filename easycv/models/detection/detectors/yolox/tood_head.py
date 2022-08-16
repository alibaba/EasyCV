import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init

from easycv.models.builder import HEADS
from .yolo_head_template import YOLOXHead_Template


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs=6,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat


@HEADS.register_module
class TOODHead(YOLOXHead_Template):

    def __init__(
            self,
            num_classes,
            model_type='s',
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act='silu',
            conv_type='conv',
            stage='CLOUD',
            obj_loss_type='BCE',
            reg_loss_type='giou',
            stacked_convs=3,
            la_down_rate=32,
            decode_in_inference=True,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
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
        super(TOODHead, self).__init__(
            num_classes=num_classes,
            model_type=model_type,
            strides=strides,
            in_channels=in_channels,
            act=act,
            conv_type=conv_type,
            stage=stage,
            obj_loss_type=obj_loss_type,
            reg_loss_type=reg_loss_type,
            decode_in_inference=decode_in_inference)

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feat_channels = int(256 * self.width)

        self.cls_decomps = nn.ModuleList()
        self.reg_decomps = nn.ModuleList()

        self.inter_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            self.cls_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate,
                                  self.conv_cfg, self.norm_cfg))
            self.reg_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate,
                                  self.conv_cfg, self.norm_cfg))

        for i in range(self.stacked_convs):
            conv_cfg = self.conv_cfg
            chn = self.feat_channels

            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_decomp, reg_decomp, cls_conv, reg_conv, stride_this_level,
                x) in enumerate(
                    zip(self.cls_decomps, self.reg_decomps, self.cls_convs,
                        self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)

            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_x = cls_decomp(feat, avg_feat)
            reg_x = reg_decomp(feat, avg_feat)

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
