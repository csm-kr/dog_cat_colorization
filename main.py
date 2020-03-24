import argparse
import train as tr
import test as te
import torch
import visdom
from dataset import ColorDataset
import torchvision


if __name__ == "__main__":
    # 1. parser 설정하기
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_folder', type=str, default='samples')
    parser.add_argument('--save_folder', type=str, default='saves')
    parser.add_argument('--train_path', type=str, default='./data/train')
    parser.add_argument('--test_path', type=str, default='./data/test')
    opts = parser.parse_args()
    print(opts)

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom()

    # 4. dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256))
    ])

    train_set = ColorDataset(root='D:\Data\dogs-vs-cats', subset='train', transform=transform)
    test_set = ColorDataset(subset='test', transform=transform)

    # 5. dataloader
    

    # train
    # tr.train(opts=opts, is_visdom=True, pretrain=True)
    # test
    te.test(opts=opts)





