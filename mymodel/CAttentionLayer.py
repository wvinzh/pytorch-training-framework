from torch import nn


class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)
        return x*y
