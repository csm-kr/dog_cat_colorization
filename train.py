import torch
import os
import time


def train(epoch, device, vis, data_loader, model, criterion, optimizer, save_path, save_file_name):

    model.train()
    tic = time.time()
    print('Epoch : {}'.format(epoch + 1))

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

        if idx % 1 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'learning rate: {lr:.7f} s \t'
                  'Time: {time:.4f} s \t'
                  .format(epoch + 1, idx, len(data_loader),
                          loss=loss.item(),
                          lr=lr,
                          time=toc))
        # ----- save -----

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_file_name + '.{}.pth'.format(epoch + 1)))








