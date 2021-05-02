import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(nin, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(nout, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True)
                                         )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(2),
                                       DoubleConv(nin, nout))

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(nin, nout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, nin, nout):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.in_conv = DoubleConv(nin, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 //  2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, nout)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    img = torch.rand([10, 1, 256, 256]).cuda()
    model = UNet(nin=1, nout=2).cuda()
    print(model.forward(img).size())

    # x5 is 32, 32

    # net = UNet().cuda()
    # print(net.forward(x).size())  # x5 is 32, 32
    # model = EDNet().cuda()
