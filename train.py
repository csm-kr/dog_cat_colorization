import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import loader
import net as network
import torch.nn as nn
import torch.optim as optim
import visdom
import time


def train(opts, is_visdom=False, pretrain=False):
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_visdom:
        # visdom 사용하기...
        vis = visdom.Visdom()

    # parser 확인
    print('epoch :', opts.epoch)

    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), ])

    # data set 정의
    train_set = loader.ColorLoader(root=opts.train_path, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    # model 정의
    net = network.Net().to(device)
    if pretrain:
        net.load_state_dict(torch.load('./saves/color.ckpt'))
        print('using pre-trained model...')

    # loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opts.lr)

    # train
    total_step = len(train_loader)

    for epoch in range(opts.epoch):

        running_loss = 0.0
        epoch_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)

            if is_visdom:
                # vis update 하는 부분...
                vis.line(X=torch.ones((1, 1)).cpu() * i + epoch * train_set.__len__() / opts.batch_size,
                         Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                         win='loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Loss'])
                         )

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}'
                      .format(epoch + 1, opts.epoch, i + 1, total_step, loss.item(), time.time() - epoch_time))

        torch.save(net.state_dict(), opts.save_folder + '/color{}.ckpt'.format(101+epoch))


    # torch.save(net.state_dict(), opts.sample_folder + '/color200.ckpt')







