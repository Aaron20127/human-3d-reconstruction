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


def save_model(path, model, epoch):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    torch.save(data, path)


def load_model(model_path, model):
    if not os.path.exists(model_path):
        print('no model will be loaded.')
        return

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    # model.load_state_dict(state_dict_, strict=False)

    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('model') and not k.startswith('model_list'):
            state_dict[k[6:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


