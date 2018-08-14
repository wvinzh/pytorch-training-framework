from torch import nn


class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super(SpatialAttentionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.sigmoid(y)
        x = x*y
        return x
