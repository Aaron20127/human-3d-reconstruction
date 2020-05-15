from .opts import opt
import numpy as np
import torch
import torch.nn as nn
from .util import gather_feat, transpose_and_gather_feat, sigmoid

def _nms(heat, kernel=3):
    ''' find the heat map center.'''
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=32):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=32):
    batch, cat, height, width = scores.size()
    # get first k value of points from every class
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float() # row in heat map
    topk_xs   = (topk_inds % width).int().float() # column in heat map
    # get first k sroces of points from the rest of all class
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode(output, thresh=0.2, down_ratio=4.0):
    hm, wh, cd, pose, shape, camera = \
        output['box_hm'], output['box_wh'], output['box_cd'], \
        output['pose'], output['shape'], output['camera']

    b, c, h, w = hm.size()

    # heat = sigmoid(hm)
    score = _nms(hm) # get score map

    mask = (score > thresh).view(b, h, w)
    ret = []

    for i in range(b):
        wh_ = wh[i, :, mask[i]].T
        cd_ = cd[i, :, mask[i]].T
        pose_ = pose[i, :, mask[i]].T
        shape_ = shape[i, :, mask[i]].T
        camera_ = camera[i, :, mask[i]].T
        center_ = mask[i].nonzero().type(torch.float32)
        score_ = score[i, :, mask[i]][0]

        if len(center_) > 0:
            # bbox
            c = center_.clone()
            c[:, 0] = center_[:, 1]
            c[:, 1] = center_[:, 0]

            lt = ((c + cd_)* down_ratio - wh_ / 2.0)
            rb = ((c + cd_)* down_ratio + wh_ / 2.0)

            bbox_ = torch.cat((lt, rb), 1)

            # camera
            # c = (c + cd_ + camera_[:, :2]) * down_ratio
            # f = (camera_[:, 2].abs() * torch.sqrt(wh_[:, 0] * wh_[:, 1])).view(-1,1)

            c = (c + cd_) * down_ratio
            f = (torch.sqrt(wh_[:, 0] * wh_[:, 1])).view(-1,1) * 4

            k = torch.eye(4, 4).unsqueeze(0).expand(c.size(0), 4, 4)
            k[:, 0, 0] = f[:, 0]
            k[:, 1, 1] = f[:, 0]
            k[:, 0, 2] = c[:, 0]
            k[:, 1, 2] = c[:, 1]
            k[:, 2, 3] = opt.camera_pose_z

            ret.append({
                'score': score_.detach().cpu().numpy(),
                'bbox': bbox_.detach().cpu().numpy(),
                'pose': pose_.detach().cpu().numpy(),
                'shape': shape_.detach().cpu().numpy(),
                'camera': k.detach().cpu().numpy()})
        else:
            ret.append({})

    return ret


def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_joints = kps.shape[1] // 2
  # heat = torch.sigmoid(heat) # This already been used in train loss phase.
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _transpose_and_gather_feat(kps, inds) # vector of key points to object center
  kps = kps.view(batch, K, num_joints * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints) # vector + center  = key points position
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints) # vector + center  = key points position

  # add decimal to object center
  if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5

  wh = _transpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)

  ## human key points heat map
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1 ## key points thresh
      kps = kps.view(batch, K, num_joints, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)

      ## add decamail to key points
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _transpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_joints, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5

      # area of none mask will be negative
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs

      # key points can be obtained from two way, one is hm_hp, another is hm+hps
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_joints, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
          batch, num_joints, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_joints, K, 2)

      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)

      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_joints, K, 2)
      kps = (1 - mask) * hm_kps + mask * kps # kps is used inside the bbox, hm_kps is used outside the box
      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_joints * 2)
  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections