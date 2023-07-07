import torch
import torch.nn as nn
from .attention import ChannelAttention, SpatialAttention
from math import sqrt



class VDSR(nn.Module):
    def __init__(self, num_channels, base_channels, num_residuals):
        super(VDSR, self).__init__()

        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True))
        self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)) for _ in range(num_residuals)])
        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def weight_init(self):
        # for m in self._modules:
        _initialize_weights(self)

    def forward(self, x):
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        return x


class VDSRAttention(nn.Module):
    def __init__(self, num_channels, base_channels, num_residuals):
        super(VDSRAttention, self).__init__()

        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), 
                                        nn.ReLU(inplace=True),
                                        )
        residual_layers = []
        for idx in range(num_residuals):
            if idx != 0 and idx % 4 == 0:
                residual_layers.append(nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), 
                                                             nn.ReLU(inplace=True),
                                                             ChannelAttention(base_channels, 8),
                                                             SpatialAttention(7)))
            else:
                residual_layers.append(nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), 
                                                                nn.ReLU(inplace=True)))
        
        self.residual_layers = nn.Sequential(*residual_layers)
        
        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def weight_init(self):
        # for m in self._modules:
        _initialize_weights(self)

    def forward(self, x):
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        return x



def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))