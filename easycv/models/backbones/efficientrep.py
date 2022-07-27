from torch import nn
from easycv.models.backbones.yolo6_blocks import RepVGGBlock, RepBlock, SimSPPF
from torchsummaryX import summary
import math
import torch


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1]
            )
        )

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
            )
        )

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3]
            )
        )

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4]
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)

if __name__=='__main__':
    depth_mul = 0.33
    width_mul = 0.5
    num_repeat_backbone = [1, 6, 12, 18, 6]
    channels_list_backbone = [64, 128, 256, 512, 1024]
    num_repeat_neck = [12, 12, 12, 12]
    channels_list_neck = [256, 128, 128, 256, 256, 512]

    channels = 3

    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in
                  (num_repeat_backbone + num_repeat_neck)]

    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
    model = EfficientRep(in_channels=channels, channels_list=channels_list, num_repeats=num_repeat)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    model = model.cuda()

    a = torch.randn(1,3,640,640).cuda()
    summary(model,a)

