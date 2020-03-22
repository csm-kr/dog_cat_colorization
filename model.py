import torch
import torch.nn as nn
from collections import OrderedDict


# 각 layer initialization
def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

        # for name, module in self.decoder.named_children():
        #     print("name :", name)
        #     print("module :", module)
        # print(x.size())

        return x


if __name__ == "__main__":
    x = torch.rand([10, 1, 224, 224]).cuda()
    net = Net().cuda()
    net.forward(x)



