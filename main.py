import argparse
import torch
import visdom
from dataset import ColorDataset
import torchvision
from torch.utils.data import DataLoader
from model import UNet, EDNet
import torch.nn as nn
import torch.optim as optim
from train import train


if __name__ == "__main__":
    # 1. parser 설정하기
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='ednet')

    opts = parser.parse_args()
    print(opts)

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom()

    # 4. data set
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256))
    ])

    train_set = ColorDataset(root='D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages', subset='train', transform=transform)
    # test_set = ColorDataset(subset='test', transform=transform)

    # 5. data loader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              shuffle=True)

    # test_loader = DataLoader(dataset=test_set,
    #                          batch_size=opts.batch_size,
    #                          shuffle=True)

    # 6. model
    model = EDNet().to(device)

    # 7. criterion
    criterion = nn.MSELoss()

    # 8. optimizer
    optimizer = optim.Adam(params=model.parameters(),
                           lr=opts.lr,
                           weight_decay=1e-5)

    print(optimizer.state_dict())

    # ----------------- for ------------------

    # for epoch in range(opts.epoch):
    #     # 9. train
    #     train(epoch=epoch,
    #           device=device,
    #           vis=vis,
    #           data_loader=train_loader,
    #           model=model,
    #           criterion=criterion,
    #           optimizer=optimizer,
    #           save_path=opts.save_path,
    #           save_file_name=opts.save_file_name)

        # 10. test
        # test

    # te.test(opts=opts)

    # ---------------------- review ----------------------
    # 1. argparse
    # 2. device
    # 3. visdom
    # 4. dataset
    # 5. dataloader
    # 6. model
    # 7. criterion
    # 8. optimizer
    # ----- for -----
    # 9. train
    # 10. test





