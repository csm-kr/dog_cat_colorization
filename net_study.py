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
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=28,
                                          out_channels=14,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          output_padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            # outsize : (112, 112, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            # outsize : (112, 112, 128)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            # outsize : (56, 56, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            # outsize : (56, 56, 256)
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            # outsize : (28, 28, 256)
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            # outsize : (28, 28, 512)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # channel 만 줄일 때
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            # outsize : (28, 28, 256)
            nn.ReLU(),
            #
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # outsize : (56, 56, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            # outsize : (56, 56, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
            # outsize : (112, 112, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            # outsize : (112, 112, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, 3, 2, 1, 1),
            # outsize : (224, 224, 2)
        )
        self.apply(weights_init)

    def forward(self, x):

        x = self.encoder(x)
        for name, module in self.encoder.named_children():
            print(name, module)
        x = self.decoder(x)
        for name, module in self.decoder.named_children():
            print(name, module)

        print(x.size())

        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True))),
            ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU(inplace=True))),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        # .named_children() 한번 확인해보자

        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        params = OrderedDict()

        for name, module in self.layers.named_children():
            print("name :", name)
            print("module :", module)
            for child in module.children():
                print("child :", child)
                for k, p in child._parameters.items():
                    # 만약 child 즉 module 의 child module 이 Conv2d 등 weight 와 bias 가 필요한 것이면
                    # child 의 _parameters 는 orderedDict()로 되어있는데,
                    # k : {weight, bias} 둘 중 하나로 되어있고
                    # P : 는 실제 tensor 값이 존재한다.
                    print("k :", k)
                    print("p :", p)

        return x


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            print(k, p)
            if p is None:
                continue
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=28,
                                          out_channels=14,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          output_padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            # outsize : (112, 112, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            # outsize : (112, 112, 128)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            # outsize : (56, 56, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            # outsize : (56, 56, 256)
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            # outsize : (28, 28, 256)
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            # outsize : (28, 28, 512)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # channel 만 줄일 때
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            # outsize : (28, 28, 256)
            nn.ReLU(),
            #
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # outsize : (56, 56, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            # outsize : (56, 56, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
            # outsize : (112, 112, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            # outsize : (112, 112, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, 3, 2, 1, 1),
            # outsize : (224, 224, 2)
        )
        self.apply(weights_init)

    def forward(self, x):

        x = self.encoder(x)
        for name, module in self.encoder.named_children():

            print("name :", name)
            print("module :", module)

            # module.weight : Parameter containing: tensor([[[[ 1.7421e-02,  8.2424e-02, -9.0914e-02], ...
            # Parameter containing: 이라고 컨테이너가 나옴
            print("module.weight :", module.weight)

            # module.weight.data : tensor([[[[ 3.5602e-02, -2.6037e-02,  8.3296e-02], ... 바로 tensor 가 나옴
            print("module.weight.data :", module.weight.data)

            # 얘는 OrderedDict() 가 나옴 : key 가 {weight, bias}, 이고 value 가 tensor([[[[ 3.5602e-02,  ...
            print("module._parameters :", module._parameters)

        x = self.decoder(x)
        # for name, module in self.decoder.named_children():
        #     print(name, module)

        print(x.size())

        return x

if __name__ == "__main__":
    x = torch.rand([10, 1, 224, 224]).cuda()
    net = MDNet().cuda()
    net.forward(x)



