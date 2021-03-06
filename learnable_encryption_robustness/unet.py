"""
ref: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
"""

import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            # changed from ref impl
            # TODO: should it add x to x1?
            x1 = self.conv(x + x1)
        return x1


class RRCnnBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCnnBlock, self).__init__()
        self.rcnn = nn.Sequential(
            RecurrentBlock(ch_out, t=t), RecurrentBlock(ch_out, t=t)
        )
        self.Conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv1x1(x)
        x1 = self.rcnn(x)
        return x + x1


class SingleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)

        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)

        x4 = self.max_pool(x3)
        x4 = self.conv4(x4)

        x5 = self.max_pool(x4)
        x5 = self.conv5(x5)

        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)
        return d1


class R2UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(R2UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.rrcnn1 = RRCnnBlock(ch_in=img_ch, ch_out=64, t=t)
        self.rrcnn2 = RRCnnBlock(ch_in=64, ch_out=128, t=t)
        self.rrcnn3 = RRCnnBlock(ch_in=128, ch_out=256, t=t)
        self.rrcnn4 = RRCnnBlock(ch_in=256, ch_out=512, t=t)
        self.rrcnn5 = RRCnnBlock(ch_in=512, ch_out=1024, t=t)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.up_rrcnn5 = RRCnnBlock(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.up_rrcnn4 = RRCnnBlock(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.up_rrcnn3 = RRCnnBlock(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_rrcnn2 = RRCnnBlock(ch_in=128, ch_out=64, t=t)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.rrcnn1(x)

        x2 = self.max_pool(x1)
        x2 = self.rrcnn2(x2)

        x3 = self.max_pool(x2)
        x3 = self.rrcnn3(x3)

        x4 = self.max_pool(x3)
        x4 = self.rrcnn4(x4)

        x5 = self.max_pool(x4)
        x5 = self.rrcnn5(x5)

        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        d1 = self.conv_1x1(d2)
        return d1


class AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(AttUNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)

        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)

        x4 = self.max_pool(x3)
        x4 = self.conv4(x4)

        x5 = self.max_pool(x4)
        x5 = self.conv5(x5)

        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)
        return d1


class R2AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(R2AttUNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.rrcnn1 = RRCnnBlock(ch_in=img_ch, ch_out=64, t=t)
        self.rrcnn2 = RRCnnBlock(ch_in=64, ch_out=128, t=t)
        self.rrcnn3 = RRCnnBlock(ch_in=128, ch_out=256, t=t)
        self.rrcnn4 = RRCnnBlock(ch_in=256, ch_out=512, t=t)
        self.rrcnn5 = RRCnnBlock(ch_in=512, ch_out=1024, t=t)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_rrcnn5 = RRCnnBlock(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_rrcnn4 = RRCnnBlock(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_rrcnn3 = RRCnnBlock(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_rrcnn2 = RRCnnBlock(ch_in=128, ch_out=64, t=t)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.rrcnn1(x)

        x2 = self.max_pool(x1)
        x2 = self.rrcnn2(x2)

        x3 = self.max_pool(x2)
        x3 = self.rrcnn3(x3)

        x4 = self.max_pool(x3)
        x4 = self.rrcnn4(x4)

        x5 = self.max_pool(x4)
        x5 = self.rrcnn5(x5)

        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        d1 = self.conv_1x1(d2)
        return d1