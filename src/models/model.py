import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from losses import FocalLoss, L1loss, L2loss, pose_l2_loss, shape_l2_loss, kp2d_l1_loss
from utils.util import batch_orth_proj, Rx_mat, Ry_mat, Rz_mat, transpose_and_gather_feat
from utils.opts import opt
from network.dla import DlaSeg
from network.smpl import SMPL


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        self.smpl = SMPL(opt.smpl_path)
        print('finished create smpl module.')


    def forward(self, output, batch):
        hm_loss, wh_loss, cd_loss, pose_loss, shape_loss, kp2d_loss = 0., 0., 0., 0., 0., 0.

        ## 1.loss of object bbox
        # heat map loss of objects center
        box_hm_cp = output['box_hm'].detach().clone()
        hm_loss = FocalLoss(output['box_hm'], batch['box_hm'])

        print('box_hm: {}'.format(torch.abs(box_hm_cp-output['box_hm']).sum()))

        # bbox heigt and lenghth
        wh_loss = L1loss(output['box_wh'], batch['box_mask'],
                                 batch['box_ind'], batch['box_wh'])


        # bbox center decimal loss
        cd_loss = L1loss(output['box_cd'], batch['box_mask'],
                                 batch['box_ind'], batch['box_cd'])


        ## 2. loss of pose and shape
        if opt.batch_size_hum36m > 0:
            pose_loss = pose_l2_loss(output['pose'], batch['theta_mask'],
                                       batch['box_ind'], batch['has_theta'], batch['pose'])


            shape_loss = shape_l2_loss(output['shape'], batch['theta_mask'],
                                         batch['box_ind'], batch['has_theta'], batch['shape'])
        else:
            pose_loss = torch.tensor(0.).to(opt.device)
            shape_loss = torch.tensor(0.).to(opt.device)


        ## 3. loss of key points
        kp2d = self._get_pred_kp2d(output['pose'], output['shape'], output['camera'],
                                   batch['box_ind'], batch['kp2d_mask'])
        kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'])


        ## total loss
        loss = opt.hm_weight * hm_loss + \
               opt.wh_weight * wh_loss + \
               opt.cd_weight * cd_loss + \
               opt.pose_weight * pose_loss + \
               opt.shape_weight * shape_loss + \
               opt.kp2d_weight * kp2d_loss

        loss_stats = {'loss': loss,
                      'hm': hm_loss,
                      'wh': wh_loss,
                      'cd': cd_loss,
                      'pose': pose_loss,
                      'shape': shape_loss,
                      'kp2d': kp2d_loss}

        return loss, loss_stats


    def _get_pred_kp2d(self, pose, shape, camera, ind, mask):
        # pose = pose[has_theta.flatten()==1, ...]
        # shape = shape[has_theta.flatten()==1, ...]
        # camera = camera[has_theta.flatten()==1, ...]
        # ind = ind[has_theta.flatten() == 1, ...]

        pred = transpose_and_gather_feat(pose, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        pose = pred[mask_pre == 1].view(-1, 72)

        pred = transpose_and_gather_feat(shape, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        shape = pred[mask_pre == 1].view(-1, 10)

        pred = transpose_and_gather_feat(camera, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        camera = pred[mask_pre == 1].view(-1, 3)

        ## smpl
        _, kp3d, _, _ = self.smpl(beta=shape, theta=pose)

        ## globle rotation
        # R = torch.matmul(Rz_mat(camera[:, 5]), \
        #                  torch.matmul(Ry_mat(camera[:, 4]), Rx_mat(camera[:, 3])))
        # kp3d = torch.matmul(kp3d, R.permute(0,2,1))

        kp2d = batch_orth_proj(kp3d, camera) # TODO fisrt tranlation or first scale ?

        return kp2d


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss


    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_states = self.loss(outputs, batch)
        return outputs, loss, loss_states


class HmrNetBase(nn.Module):
    def __init__(self):
        super(HmrNetBase, self).__init__()

        print('start creating sub modules...')
        self._create_sub_modules()


    def _create_sub_modules(self):
        self.encoder = DlaSeg(opt.heads,
                              not_use_dcn=not opt.use_dcn)
        print('finished create encoder module.')

        # self.smpl = SMPL(opt.smpl_path)
        # print('finished create smpl module.')


    def forward(self, input):
        ## encoder
        ret = self.encoder(input)

        ## smpl
        # b, w, h, _ = ret['camera'].size()
        # camera = ret['camera'].view(-1,6) # (batch*128*128, 3)
        # pose = ret['pose'].permute(0,2,3,1).view(-1, 72)
        # shape = ret['shape'].permute(0,2,3,1).view(-1, 10)
        #
        # j3d = self.smpl(beta=shape, theta=pose)
        # j2d = batch_orth_proj(j3d, camera)
        #
        # j3d = j3d.view(b ,w, h, j3d.shape[-2], j3d.shape[-1])
        # j2d = j3d.view(b ,w, h, j2d.shape[-2], j2d.shape[-1])
        #
        # ret.update({'kp_2d': j2d, 'kp_3d':j3d})

        return ret