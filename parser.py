import argparse

# parser 설정하기
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_folder', type=str, default='samples')
parser.add_argument('--save_folder', type=str, default='saves')
parser.add_argument('--train_path', type=str, default='./data/train')
parser.add_argument('--test_path', type=str, default='./data/test')