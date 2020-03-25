import torch
import torch.nn as nn
from collections import OrderedDict


# 각 layer initialization
def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # encoder
        self.x1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )

        self.x2 = nn.Sequential(nn.MaxPool2d(2),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, 3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )

        self.x3 = nn.Sequential(nn.MaxPool2d(2),
                                nn.Conv2d(128, 256, 3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )

        self.x4 = nn.Sequential(nn.MaxPool2d(2),
                                nn.Conv2d(256, 512, 3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                )

        self.x5 = nn.Sequential(nn.MaxPool2d(2),
                                nn.Conv2d(512, 1024, 3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(1024, 1024, 3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(inplace=True),
                                )

        # decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3_conv = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      )

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4_conv = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      )

        self.final = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),
                                   )

        self.apply(weights_init)

    def forward(self, x):

        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        up1 = self.up1(x5)
        up2 = self.up1_conv(torch.cat([x4, up1], dim=1))

        up2 = self.up2(up2)
        up3 = self.up2_conv(torch.cat([x3, up2], dim=1))

        up3 = self.up3(up3)
        up4 = self.up3_conv(torch.cat([x2, up3], dim=1))

        up4 = self.up4(up4)
        up4 = self.up4_conv(torch.cat([x1, up4], dim=1))

        out = self.final(up4)  # torch.Size([5, 2, 256, 256])

        # for name, module in self.decoder.named_children():
        #     print("name :", name)
        #     print("module :", module)
        # print(x.size())

        return out


class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict(
            [
            ('conv1', nn.Sequential(nn.Conv2d(1, 64, 3, 2, 1),
                                    # outsize : (112, 112, 64)
                                    nn.ReLU(),)),
            ('conv2', nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                    # outsize : (112, 112, 128)
                                    nn.ReLU(),)),
            ('conv3', nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1),
                                    # outsize : (56, 56, 128)
                                    nn.ReLU(),)),
            ('conv4', nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                    # outsize : (56, 56, 256)
                                    nn.ReLU(),)),
            ('conv5', nn.Sequential(nn.Conv2d(256, 256, 3, 2, 1),
                                    # outsize : (28, 28, 256)
                                    nn.ReLU(),)),
            ('conv6', nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                    # outsize : (28, 28, 512)
                                    nn.ReLU(),)),
        ]
        ))

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(512, 256, 3, 1, 1),
                # outsize : (28, 28, 256)
                nn.ReLU(),)),
            ('deconv2', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
                # outsize : (56, 56, 256)
                nn.ReLU(),)),
            ('deconv3', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(256, 128, 3, 1, 1),
                # outsize : (56, 56, 128)
                nn.ReLU(),)),
            ('deconv4', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
                # outsize : (112, 112, 128)
                nn.ReLU(),)),
            ('deconv5', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(128, 64, 3, 1, 1),
                # outsize : (112, 112, 64)
                nn.ReLU(),)),
            ('deconv6', nn.Sequential(
                # channel 만 줄일 때
                nn.ConvTranspose2d(64, 2, 3, 2, 1, 1),
                # outsize : (224, 224, 2)
                )),
        ]))
        self.apply(weights_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    x = torch.rand([10, 1, 256, 256]).cuda()

    # net = UNet().cuda()
    # print(net.forward(x).size())  # x5 is 32, 32

    model = EDNet().cuda()
    print(model.forward(x).size())  # x5 is 32, 32



