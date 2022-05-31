import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register_module
class BenchMarkMLP(nn.Module):

    def __init__(self,
                 feature_num,
                 num_classes=1000,
                 avg_pool=False,
                 **kwargs):
        super(BenchMarkMLP, self).__init__()

        self.fc1 = nn.Linear(feature_num, feature_num)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = avg_pool

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if self.avg_pool:
            x = self.pool(x)
        x = self.fc1(x)
        x = self.relu1(x)
        return tuple([x])
