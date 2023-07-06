from torch import nn
from .attention import ChannelAttention, SpatialAttention


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class SRCNNAttention(nn.Module):

    def __init__(self, num_channels=3):
        super(SRCNNAttention, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)

        self.channel_attention1 = ChannelAttention(64, 8)
        self.spatial_attention1 = SpatialAttention(7)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)

        self.channel_attention2 = ChannelAttention(32, 8)
        self.spatial_attention2 = SpatialAttention(7)

        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.channel_attention1(x)
        x = self.spatial_attention1(x)
        x = self.relu(self.conv2(x))
        x = self.channel_attention2(x) 
        x = self.spatial_attention2(x) 
        x = self.conv3(x)

        return x
    
    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)



def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1:
        if m.bias is None:
            nn.init.kaiming_normal_(m.weight)