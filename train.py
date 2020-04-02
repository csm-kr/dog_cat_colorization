import torch
import os
import time
import numpy as np
from skimage import color


def train(epoch, device, vis, data_loader, model, criterion, optimizer, save_path, save_file_name):

    model.train()
    tic = time.time()
    print('Epoch : {}'.format(epoch))

    for idx, (images, labels) in enumerate(data_loader):

        # ----- cuda -----
        images = images.to(device)
        labels = labels.to(device)

        # ----- loss -----
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ----- update -----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        toc = time.time() - tic

        # ----- print -----

        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        # print
        vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * data_loader.__len__(),
                 Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                 win='loss',
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='Loss',
                           title='train loss',
                           legend=['Loss'])
                 )

        # visdom image plotting
        vis_gray_img = images[0].cpu().numpy()  # [1, H, W]
        vis_ab_img = labels[0].cpu().numpy()  # [2, H, W]
        vis_out_img = outputs[0].detach().cpu().numpy()  # [2, H, W]

        # [C, H, W] --> [H, W, C]
        vis_gray_img = vis_gray_img.transpose((1, 2, 0))
        vis_ab_img = vis_ab_img.transpose((1, 2, 0))
        vis_out_img = vis_out_img.transpose((1, 2, 0))

        # lab to rgb
        # 1) de-normalization
        vis_gray_img_ = vis_gray_img * 100
        vis_ab_img = vis_ab_img * 255 - 128
        vis_out_img = vis_out_img * 255 - 128

        # 2) lab2rgb
        color_img = np.concatenate((vis_gray_img_, vis_ab_img), axis=-1)
        origin_img = color.lab2rgb(color_img)  # rgb

        label_img = np.concatenate((vis_gray_img_, vis_out_img), axis=-1)
        pred_img = color.lab2rgb(label_img)  # rgb

        # [C, H, W] --> [H, W, C] --> [C, H, w]
        vis_gray_img = vis_gray_img.transpose((2, 0, 1))
        origin_img = origin_img.transpose((2, 0, 1))
        pred_img = pred_img.transpose((2, 0, 1))

        vis.image(vis_gray_img,
                  win='gray_img',
                  opts=dict(title='gray_img'))

        vis.image(origin_img,
                  win='origin_img',
                  opts=dict(title='origin_img'))

        vis.image(pred_img,
                  win='pred_img',
                  opts=dict(title='pred_img'))

        if idx % 10 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'learning rate: {lr:.7f} s \t'
                  'Time: {time:.4f} s \t'
                  .format(epoch,
                          idx, len(data_loader),
                          loss=loss.item(),
                          lr=lr,
                          time=toc))
        # ----- save -----

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_file_name + '.{}.pth'.format(epoch)))








