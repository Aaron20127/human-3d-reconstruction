import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from losses import FocalLoss, L1loss, L2loss, pose_l2_loss, shape_l2_loss, kp2d_l1_loss, kp3d_l2_loss
from utils.util import batch_orth_proj, Rx_mat, Ry_mat, Rz_mat, transpose_and_gather_feat, sigmoid
from utils.opts import opt
from network.dla import DlaSeg
from network.smpl import SMPL


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        self.smpl = SMPL(opt.smpl_path)
        print('finished create smpl module.')


    def forward(self, output, batch):
        hm_loss, wh_loss, cd_loss, pose_loss, \
            shape_loss, kp2d_loss, kp3d_loss = torch.zeros(7).to(opt.device)

        ## 1.loss of object bbox
        # heat map loss of objects center
        output['box_hm'] = sigmoid(output['box_hm'])
        hm_loss = FocalLoss(output['box_hm'], batch['box_hm'])


        # bbox heigt and lenghth
        wh_loss = L1loss(output['box_wh'], batch['box_mask'],
                                 batch['box_ind'], batch['box_wh'])



        # bbox center decimal loss
        cd_loss = L1loss(output['box_cd'], batch['box_mask'],
                                 batch['box_ind'], batch['box_cd'])


        ## 2. loss of pose and shape
        if opt.pose_weight > 0 and 'pose' in batch:
            pose_loss = pose_l2_loss(output['pose'], batch['theta_mask'],
                                       batch['box_ind'], batch['has_theta'], batch['pose'])

        if opt.shape_weight > 0 and 'shape' in batch:
            shape_loss = shape_l2_loss(output['shape'], batch['theta_mask'],
                                         batch['box_ind'], batch['has_theta'], batch['shape'])


        ## 3. loss of key points
        if opt.kp2d_weight > 0 and 'kp2d' in batch:
            kp2d = self._get_pred_kp2d(output['pose'], output['shape'], output['camera_off'],
                                       output['box_cd'], output['box_wh'],
                                       batch['box_ind'], batch['kp2d_mask'])
            kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'])

        if opt.kp3d_weight > 0 and 'kp3d' in batch:
            kp3d = self._get_pred_kp3d(output['pose'], output['shape'], output['has_theta'],
                                  batch['box_ind'], batch['kp2d_mask'])
            kp3d_loss = kp3d_l2_loss(kp3d, batch['kp3d_mask'], batch['kp3d'])


        ## total loss
        loss = opt.hm_weight * hm_loss + \
               opt.wh_weight * wh_loss + \
               opt.cd_weight * cd_loss + \
               opt.pose_weight * pose_loss + \
               opt.shape_weight * shape_loss + \
               opt.kp2d_weight * kp2d_loss + \
               opt.kp3d_weight * kp3d_loss


        loss_stats = {'loss': loss,
                      'hm': hm_loss,
                      'wh': wh_loss,
                      'cd': cd_loss,
                      'pose': pose_loss,
                      'shape': shape_loss,
                      'kp2d': kp2d_loss,
                      'kp3d': kp3d_loss}

        return loss, loss_stats


    def _get_pred_kp2d(self, pose, shape, camera_off, cd, wh, ind, mask):
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


        ## smpl
        _, kp3d, _, _ = self.smpl(beta=shape, theta=pose)


        ## kp2d
        ind_ = ind[mask == 1].view(ind.size(0), -1)
        box_center = torch.cat((ind_ % opt.output_res, ind_ // opt.output_res), 1)

        pred = transpose_and_gather_feat(wh, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_wh = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(cd, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_cd = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(camera_off, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        camera_off = pred[mask_pre == 1].view(-1, 3)

        c = (box_center + box_cd) * opt.down_ratio
        f = torch.sqrt(box_wh[:,0] * box_wh[:,1])

        ## globle rotation
        # R = torch.matmul(Rz_mat(camera[:, 5]), \
        #                  torch.matmul(Ry_mat(camera[:, 4]), Rx_mat(camera[:, 3])))
        # kp3d = torch.matmul(kp3d, R.permute(0,2,1))

        kp2d = batch_orth_proj(kp3d, camera) # TODO fisrt tranlation or first scale ?

        return kp2d


    def _get_pred_kp3d(self, pose, shape, has_kp3d, ind, mask):
        # pose = pose[has_theta.flatten()==1, ...]
        # shape = shape[has_theta.flatten()==1, ...]
        # camera = camera[has_theta.flatten()==1, ...]
        # ind = ind[has_theta.flatten() == 1, ...]
        pose = pose[(has_kp3d==1).flatten(), ...]
        shape = shape[(has_kp3d==1).flatten(), ...]
        ind = ind[(has_kp3d==1).flatten(), ...]


        pred = transpose_and_gather_feat(pose, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        pose = pred[mask_pre == 1].view(-1, 72)

        pred = transpose_and_gather_feat(shape, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        shape = pred[mask_pre == 1].view(-1, 10)


        ## smpl
        _, kp3d, _, _ = self.smpl(beta=shape, theta=pose)

        return kp3d


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