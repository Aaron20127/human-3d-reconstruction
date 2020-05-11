# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# -----------------------------------------------------------------------------
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from utils.util import transpose_and_gather_feat, sigmoid


def FocalLoss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred = sigmoid(pred)

  pos_inds = gt.eq(1).float() # the center of object is 1, others is 0
  neg_inds = gt.lt(1).float() # except for object center, everything else is 1

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # loss of center points key points heat map
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds # loss of other points

  num_pos  = pos_inds.float().sum() # number of key
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos # average loss for every object
  return loss



def L1loss(output, mask, ind, target):
    pred = transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-8)
    return loss



def L2loss(output, mask, ind, target):
    pred = transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.mse_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-8)
    return loss



def pose_l2_loss(output, mask, ind, has_theta, target):
    output = output[has_theta.flatten() == 1, ...]
    ind = ind[has_theta.flatten() == 1, ...]
    loss = L2loss(output[:, 3:, :, :], mask, ind, target[:, :, 3:])
    return loss



def shape_l2_loss(output, mask, ind, has_theta, target):
    output = output[has_theta.flatten() == 1, ...]
    ind = ind[has_theta.flatten() == 1, ...]
    loss = L2loss(output, mask, ind, target)
    return loss



def kp2d_l1_loss(output, mask, target):
    output = output.view(-1, 2)
    target = target[mask == 1, ...].view(-1, 3)

    output = output[target[:, 2] == 1]
    target = target[target[:, 2] == 1]

    loss = F.mse_loss(target[:, 0:2], output, reduce='sum')
    loss = loss / (output.size(0) * output.size(1) + 1e-8)

    return loss
