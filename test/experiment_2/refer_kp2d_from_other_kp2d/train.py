from torch.utils.data import DataLoader
# imports
import matplotlib.pyplot as plt
import numpy as np

import cv2
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import time
from torch.utils.tensorboard import SummaryWriter
from dataset import crowdpose
from model_exp_2 import NetLoss, BaseNet, ModelWithLoss, initialize_weights
from opts import opt

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../../src')

from utils.debugger import Debugger
from utils.util import conver_crowdpose_to_cocoplus

root_path = abspath + '/../../../'

debugger = Debugger(root_path + 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                    root_path + 'data/neutral_smpl_with_cocoplus_reg.pkl',
                    'cocoplus')

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

def total_parameters(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: %d , Trainable: %d' % (total_num, trainable_num))


def debug(output, batch, save_dir, id, debug_level=0):

    imagepath = batch['imagepath'][0]
    img = cv2.imread(imagepath)

    h, w = img.shape[0:2]

    # 1.
    input_id = 'kp2d_input'
    debugger.add_img(img, img_id=input_id)

    for kp2d in batch['gt'][0]['kp2d_input'][0]:
        kp2d = kp2d.detach().cpu().numpy()
        kp2d[:, 0] = kp2d[:, 0] * w
        kp2d[:, 1] = kp2d[:, 1] * h

        kp2d = conver_crowdpose_to_cocoplus(kp2d)
        debugger.add_kp2d(kp2d, img_id=input_id)

    # 2.kp2d_gt
    kp2d_gt_id = 'kp2d_gt'
    debugger.add_img(img, img_id=kp2d_gt_id)

    kp2d = batch['gt'][0]['kp2d'][0].detach().cpu().numpy()
    kp2d[:, 0] = kp2d[:, 0] * w
    kp2d[:, 1] = kp2d[:, 1] * h

    kp2d = conver_crowdpose_to_cocoplus(kp2d)
    debugger.add_kp2d(kp2d, img_id=kp2d_gt_id)


    # 3.kp2d_pre
    _output = output.view(output.size(0), -1, 2)

    kp2d_pre_id = 'kp2d_pre'
    debugger.add_img(img, img_id=kp2d_pre_id)

    kp2d = _output[0].detach().cpu().numpy()
    kp2d = np.hstack((kp2d, np.ones((kp2d.shape[0],1))))

    kp2d[:, 0] = kp2d[:, 0] * w
    kp2d[:, 1] = kp2d[:, 1] * h

    kp2d = conver_crowdpose_to_cocoplus(kp2d)
    debugger.add_kp2d(kp2d, img_id=kp2d_pre_id)

    if debug_level == 0:
        debugger.show_all_imgs(pause=True)
    elif debug_level == 1:
        debugger.save_all_imgs(id, save_dir)



## 1.log
log_id = time.strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(abspath+'/log/'+ log_id)


## 2.data loader
train_dataset = crowdpose(
    data_path=abspath+'/data',
    image_path=opt.crowdpose_path,
    split='train'
)
val_dataset = crowdpose(
    data_path=abspath+'/data',
    image_path=opt.crowdpose_path,
    split='val',
    max_data_len=5000
)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)


## 3. init network
model = BaseNet().to(opt.device)
# initialize_weights(model)

optimizer = torch.optim.Adam(model.parameters(), opt.lr)
network = ModelWithLoss(model, NetLoss())
network.train()

total_parameters(model)


##4. train
for epoch in range(opt.epoches):
    len_train_set = len(train_loader)
    len_val_set = len(val_loader)

    train_stats = {}
    if opt.val == False:
        for i, batch in enumerate(train_loader):
            for k, v in batch.items():
                if type(v) == torch.Tensor:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = network(batch)

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            for key, value in loss_stats.items():
                if key not in train_stats:
                    train_stats[key] = value.item()
                else:
                    train_stats[key] += value

            if i % 10 == 9:
                msg = "train {} / {} | epoch {} ".format(i, len_train_set, epoch)
                for key, value in train_stats.items():
                    writer.add_scalar('train_{}'.format(key),
                                      value / i,
                                      epoch * len_train_set + i)
                    msg += '| {} {:.4f} '.format(key, value / i)
                print(msg)


            if i % 100 == 99:
                value_stats = {}
                network.eval()

                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        for k, v in batch.items():
                            if type(v) == torch.Tensor:
                                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
                        output, loss, loss_stats = network(batch)

                        print('val | {} / {} '.format(j, len_val_set))
                        for key, value in loss_stats.items():
                            if key not in value_stats:
                                value_stats[key] = value
                            else:
                                value_stats[key] += value

                        debug(output['kp2d'], batch, abspath + '/log/' + log_id, j, debug_level=1)


                msg = "val | epoch {} ".format(epoch)
                for key, value in value_stats.items():
                    writer.add_scalar('val_{}'.format(key),
                                      value / len_val_set,
                                      epoch * len_train_set + i)
                    msg += '| {} {:.4f} '.format(key, value / len_val_set)
                print(msg)

                network.train()


        if opt.val == True:
            value_stats = {}
            network.eval()

            with torch.no_grad():
                for j, batch in enumerate(val_loader):
                    for k, v in batch.items():
                        if type(v) == torch.Tensor:
                            batch[k] = batch[k].to(device=opt.device, non_blocking=True)

                    output, loss, loss_stats = network(batch)

                    for key, value in loss_stats.items():
                        if key not in value_stats:
                            value_stats[key] = value
                        else:
                            value_stats[key] += value

            msg = "epoch {} ".format(epoch)
            for key, value in value_stats.items():
                writer.add_scalar('val_{}'.format(key),
                                  value / len_val_set,
                                  epoch)
                msg += '| {} {:.4f} '.format(key, value / i)
            print(msg)


print('Finished Training')

