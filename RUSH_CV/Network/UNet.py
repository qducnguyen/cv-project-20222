import torch
import torch.nn as nn

from .UNet_part import *

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)
        self.up5 = up(64, 32,bilinear=False)

        self.outc = outconv(32, n_classes)

        self.up_s1=up_s(64,32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x0 = self.up_s1(x1)
        x = self.up5(x, x0)

        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class UNet2Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2Attention, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down_attention(64, 128)
        self.down2 = down_attention(128, 256)
        self.down3 = down_attention(256, 512)
        self.down4 = down_attention(512, 512)

        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)
        self.up5 = up(64, 32,bilinear=False)

        self.outc = outconv(32, n_classes)

        self.up_s1= up_s_attention(64,32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x0 = self.up_s1(x1)
        x = self.up5(x, x0)

        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)
        self.up5 = up(64, 32,bilinear=False)

        self.outc = outconv(32, n_classes)

        self.up_s1=up_s(64,32, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x0 = self.up_s1(x1)
        x = self.up5(x, x0)

        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class UNet3Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3Attention, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down_attention(64, 128)
        self.down2 = down_attention(128, 256)
        self.down3 = down_attention(256, 512)
        self.down4 = down_attention(512, 512)

        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)
        self.up5 = up(64, 32,bilinear=False)

        self.outc = outconv(32, n_classes)

        self.up_s1=up_s_attention(64,32, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x0 = self.up_s1(x1)
        x = self.up5(x, x0)

        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)#(128, 64)
        self.up5 = up(64, 16,bilinear=False)
        self.up6 = up(32, 16,bilinear=False)
        self.outc = outconv(16, n_classes)#64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)


    def forward(self, xs):

        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x0 =self.up_s1(x1)
        x_1 =self.up_s2(x0)

        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class UNet4Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4Attention, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down_attention(64, 128)
        self.down2 = down_attention(128, 256)
        self.down3 = down_attention(256, 512)
        self.down4 = down_attention(512, 512)
        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)#(128, 64)
        self.up5 = up(64, 16,bilinear=False)
        self.up6 = up(32, 16,bilinear=False)
        self.outc = outconv(16, n_classes)#64

        self.up_s1=up_s_attention(64,32)
        self.up_s2=up_s_attention(32,16)


    def forward(self, xs):

        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x0 =self.up_s1(x1)
        x_1 =self.up_s2(x0)

        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.outc(x)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
