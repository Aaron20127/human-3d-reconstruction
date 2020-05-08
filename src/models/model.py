import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from losses import FocalLoss, RegL1Loss
from model_util import batch_orth_proj
from utils.opts import opt
from network.dla import DlaSeg
from network.smpl import SMPL


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        self.crit_hm = FocalLoss()
        # self.crit_wh = reg_loss()
        # self.crit_pose = pose_l2_loss()
        # self.crit_shape = shape_l2_loss()
        # self.crit_kp_2d = kp_2d_l1_loss()
        # self.crit_kp_3d = kp_3d_l2_loss()

    def forward(self, output, batch):
        """
            batch
            {
                'hm':           '(n, 1, 128, 128)',  # bbox center heat map

                'wh_ind':       '(n, max_obj)', # bbox width, height
                'wh_mask':      '(n, max_obj)',
                'wh':           '(n, max_obj, 2)',

                'theta_ind':    '(n, max_obj)'
                'theta_mask':   '(n, max_obj)'
                'pose':         '(n, max_obj, 72)',
                'shape':        '(n, max_obj, 10)',

                'kp_2d_ind':    '(n, max_obj)'
                'kp_3d_mask':   '(n, max_obj)'
                'kp_3d':        '(n, 19, 3)',

                'kp_2d_ind':    '(n, max_obj)'
                'kp_2d_mask':   '(n, max_obj)'
                'kp_2d':        '(n, 19, 3)', # 第三列是否可见可以作为索引，加上coco数据集的眼睛、耳朵和鼻子
            }

            output
            {
                'box_hm':       '(n, 1, 128,128)',
                'box_wh':       '(n, 2, 128,128)',
                'box_dc':       '(n, 2, 128,128)',
                'box_num':      '(n, 2, 1,  1)',

                'pose':          '(n,72,128,128)',
                'shape':         '(n,10,128,128)',
                'cam':           '(n, 3,128,128)',

                'kp2d':          '(n, 128, 128, 19, 2)'
                'kp3d':          '(n, 128, 128, 19, 3)'
            }
        """

        ## 1.loss of object bbox
        # heat map loss of objects center
        output['hm'] = _sigmoid(output['hm']) # do sigmoid
        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        # bbox heigt and lenghth
        wh_loss = self.crit_wh(output['wh'], batch['wh_mask'],
                                 batch['wh_ind'], batch['wh'])

        ## 2. loss of pose and shape
        pose_loss = self.crit_pose(output['pose'], batch['theta_mask'],
                                   batch['theta_ind'], batch['pose'])

        shape_loss = self.crit_shape(output['shape'], batch['theta_mask'],
                                     batch['theta_ind'], batch['shape'])

        ## 3. loss of key points
        kp_2d_loss = self.crit_kp_2d(output['kp_2d'], batch['kp_2d_mask'],
                                     batch['kp_2d_mask'], batch['kp_2d'])

        kp_3d_loss = self.crit_kp_3d(output['kp_3d'], batch['kp_3d_mask'],
                                     batch['kp_3d_mask'], batch['kp_3d'])

        ##
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.pose_weight * pose_loss + opt.shape_weight * shape_loss + \
               opt.kp_2d_weight * kp_2d_loss + opt.kp_3d_weight * kp_3d_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': hp_loss,
                      'pose_loss': pose_loss, 'shape_loss': shape_loss,
                      'kp_2d_loss': kp_2d_loss, 'kp_3d_loss': kp_3d_loss}
        return loss, loss_stats


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

        self.smpl = SMPL(opt.smpl_path)
        print('finished create smpl module.')


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