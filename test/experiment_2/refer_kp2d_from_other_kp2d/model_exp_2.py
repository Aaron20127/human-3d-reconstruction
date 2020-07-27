import os
import sys
import numpy as np

import torch
from opts import opt
from torch import nn


def func_kp2d_loss(output, target):
    dim0, dim1, dim2 = target.size()
    _output = output.view(dim0, dim1, 2)
    mask = target[:, :, 2] > 0

    _output = _output[mask, :]
    _target = target[mask, :][:, :2]

    loss = torch.abs(_output - _target).sum()
    loss = loss / (_output.numel() + 1e-16)

    return loss


class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        pass


    def forward(self, output, batch):

        kp2d_loss = func_kp2d_loss(output['kp2d'], batch['kp2d'])

        ## total loss
        loss = opt.kp2d_weight * kp2d_loss

        loss_stats = {'loss': loss,
                      'kp2d': kp2d_loss}

        return loss, loss_stats


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(280, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            nn.Linear(2048, 14*2)
        )

    def forward(self, x):
        kp2d = self.base(x)
        return {
            'kp2d': kp2d
        }


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss


    def forward(self, batch):
        outputs = self.model(batch['kp2d_input'])
        loss, loss_states = self.loss(outputs, batch)
        return outputs, loss, loss_states


def initialize_weights(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            param.data.normal_(mean=0, std=0.001)
        elif 'bias' in name:
            param.data.normal_(mean=0, std=0.001)