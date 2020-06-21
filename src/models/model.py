import os
import sys
import numpy as np
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from losses import FocalLoss, L1loss, L2loss, pose_l2_loss, shape_l2_loss, kp2d_l1_loss, kp3d_l2_loss, dp2d_l1_loss
from utils.util import batch_orth_proj, Rx_mat, Ry_mat, Rz_mat, transpose_and_gather_feat, sigmoid
from utils.opts import opt
from network.dla import DlaSeg
from network.smpl import SMPL


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        if opt.smpl_type == 'basic':
            self.smpl = SMPL(opt.smpl_basic_path, smpl_type=opt.smpl_type)
        elif opt.smpl_type == 'cocoplus':
            self.smpl  = SMPL(opt.smpl_cocoplus_path, smpl_type=opt.smpl_type)
        self.register_buffer('kp2d_every_weight_train', torch.tensor(opt.kp2d_every_weight_train).type(torch.float32))
        self.register_buffer('kp2d_every_weight_val', torch.tensor(opt.kp2d_every_weight_val).type(torch.float32))
        self.register_buffer('loss', torch.zeros(8).type(torch.float32))
        # self.Rx = Rx_mat(torch.tensor([np.pi])).to(opt.device)[0].T
        print('finished create smpl module.')


    def forward(self, output, batch):
        hm_loss, wh_loss, cd_loss, pose_loss, \
            shape_loss, kp2d_loss, kp3d_loss, dp2d_loss =  self.loss.zero_()

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
        # if opt.pose_weight > 0 and 'pose' in batch:
            # pose_loss = pose_l2_loss(output['pose'], batch['smpl_mask'],
            #                          batch['box_ind'], batch['has_theta'], batch['pose'],
            #                          opt.pose_loss_type)
        if opt.pose_weight > 0:
            pose_loss = pose_l2_loss(output['pose'], batch['smpl_mask'],
                                     batch['box_ind'],  batch['pose'],
                                     opt.pose_loss_type)


        # if opt.shape_weight > 0 and 'shape' in batch:
            # shape_loss = shape_l2_loss(output['shape'], batch['smpl_mask'],
            #                              batch['box_ind'], batch['has_theta'], batch['shape'])
        if opt.shape_weight > 0:
            shape_loss = shape_l2_loss(output['shape'], batch['smpl_mask'],
                                       batch['box_ind'], batch['shape'])


        ## 3. loss of key points
        # if opt.kp2d_weight > 0 and 'kp2d' in batch:
        #     if batch['kp2d_mask'].sum() > 0:
        #         kp2d = self._get_pred_kp2d(output['pose'], output['shape'], output['camera'],
        #                                    output['box_cd'], output['box_wh'],
        #                                    batch['box_ind'], batch['kp2d_mask'])
        #         if self.training:
        #             kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'], self.kp2d_every_weight_train)
        #         else:
        #             kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'], self.kp2d_every_weight_val)
        #
        # if opt.kp3d_weight > 0 and 'kp3d' in batch:
        #     if batch['kp3d_mask'].sum() > 0:
        #         kp3d = self._get_pred_kp3d(output['pose'], output['shape'], batch['has_kp3d'],
        #                                    batch['box_ind'], batch['kp3d_mask'])
        #         kp3d_loss = kp3d_l2_loss(kp3d, batch['kp3d_mask'], batch['kp3d'])
        #
        #
        # ## 4. loss of dense pose 2d
        # if opt.dp2d_weight > 0 and 'dp2d' in batch:
        #     if batch['dp_mask'].sum() > 0:
        #         dp2d = self._get_pred_dp2d(output['pose'], output['shape'], output['camera'],
        #                                      output['box_cd'], output['box_wh'],
        #                                      batch['box_ind'], batch['dp_mask'],
        #                                      batch['dp_ind'], batch['dp_rat'], batch['has_dp'])
        #         dp2d_loss = dp2d_l1_loss(dp2d, batch['dp_mask'], batch['dp2d'])


        # 3. loss of kp2d, kp3d, dp2d
        if opt.kp2d_weight > 0:
            if batch['kp2d_mask'].sum() > 0:
                kp2d = self._get_pred_kp2d(output['pose'], output['shape'], output['camera'],
                                           output['box_cd'], output['box_wh'],
                                           batch['box_ind'], batch['kp2d_mask'])
                if self.training:
                    kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'], self.kp2d_every_weight_train)
                else:
                    kp2d_loss = kp2d_l1_loss(kp2d, batch['kp2d_mask'], batch['kp2d'], self.kp2d_every_weight_val)


        if opt.kp3d_weight > 0:
            if batch['kp3d_mask'].sum() > 0:
                kp3d = self._get_pred_kp3d(output['pose'], output['shape'],
                                           batch['box_ind'], batch['kp3d_mask'])
                kp3d_loss = kp3d_l2_loss(kp3d, batch['kp3d_mask'], batch['kp3d'])


        if opt.dp2d_weight > 0:
            if batch['dp_mask'].sum() > 0:
                dp2d = self._get_pred_dp2d(output['pose'], output['shape'], output['camera'],
                                             output['box_cd'], output['box_wh'],
                                             batch['box_ind'], batch['dp_mask'],
                                             batch['dp_ind'], batch['dp_rat'])
                dp2d_loss = dp2d_l1_loss(dp2d, batch['dp_mask'], batch['dp2d'])


        ## total loss
        loss = opt.hm_weight * hm_loss + \
               opt.wh_weight * wh_loss + \
               opt.cd_weight * cd_loss + \
               opt.pose_weight * pose_loss + \
               opt.shape_weight * shape_loss + \
               opt.kp2d_weight * kp2d_loss + \
               opt.kp3d_weight * kp3d_loss + \
               opt.dp2d_weight * dp2d_loss


        loss_stats = {'loss': loss,
                      'hm': hm_loss,
                      'wh': wh_loss,
                      'pose': pose_loss,
                      'shape': shape_loss,
                      'kp2d': kp2d_loss,
                      'kp3d': kp3d_loss,
                      'dp2d': dp2d_loss,
                      'cd': cd_loss}

        return loss, loss_stats


    def _get_pred_kp2d(self, pose, shape, camera, cd, wh, ind, mask):
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
        _, kp3d, _ = self.smpl(beta=shape, theta=pose)


        ## kp2d
        ind_ = ind[mask == 1].view(-1, 1)
        box_center = torch.cat((ind_ % opt.output_res, ind_ // opt.output_res), 1).type(torch.float32)

        pred = transpose_and_gather_feat(wh, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_wh = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(cd, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_cd = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(camera, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        camera = pred[mask_pre == 1].view(-1, 3)

        c = (box_center + box_cd + camera[:, 1:]) * opt.down_ratio
        f = (camera[:, 0].abs() * torch.sqrt(box_wh[:,0].abs() * box_wh[:,1].abs()) * opt.down_ratio * opt.camera_pose_z).view(-1,1) # TODO give camera off bias a initial value

        # kp3d = torch.matmul(kp3d, self.Rx) # global rotation
        kp3d[:,:, 2] = kp3d[:,:, 2] + opt.camera_pose_z # let z be positive

        kp3d = kp3d / torch.unsqueeze(kp3d[:,:,2], 2) # homogeneous vector
        kp2d = kp3d[:,:,:2] * f.view(-1,1,1) + c.view(-1,1,2) # camera transformation

        return kp2d


    def _get_pred_kp3d(self, pose, shape, ind, mask):
        # pose = pose[has_theta.flatten()==1, ...]
        # shape = shape[has_theta.flatten()==1, ...]
        # camera = camera[has_theta.flatten()==1, ...]
        # ind = ind[has_theta.flatten() == 1, ...]
        # pose = pose[(has_kp3d==1).flatten(), ...]
        # shape = shape[(has_kp3d==1).flatten(), ...]
        # ind = ind[(has_kp3d==1).flatten(), ...]

        pred = transpose_and_gather_feat(pose, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        pose = pred[mask_pre == 1].view(-1, 72)

        pred = transpose_and_gather_feat(shape, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        shape = pred[mask_pre == 1].view(-1, 10)


        ## smpl
        _, joints, _ = self.smpl(beta=shape, theta=pose)

        return joints


    def _get_pred_dp2d(self, pose, shape, camera, cd, wh, ind, mask, dp_ind, dp_rat):
        # pose = pose[has_dp.flatten()==1, ...]
        # shape = shape[has_dp.flatten()==1, ...]
        # camera = camera[has_dp.flatten()==1, ...]
        # cd = cd[has_dp.flatten()==1, ...]
        # wh = wh[has_dp.flatten()==1, ...]
        # ind = ind[has_dp.flatten() == 1, ...]


        pred = transpose_and_gather_feat(pose, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        pose = pred[mask_pre == 1].view(-1, 72)

        pred = transpose_and_gather_feat(shape, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        shape = pred[mask_pre == 1].view(-1, 10)


        ## smpl
        verts, _, _ = self.smpl(beta=shape, theta=pose)

        ## get verts of dense pose
        dp_ind = dp_ind[mask == 1]
        dp_rat = dp_rat[mask == 1]

        dp_ind = dp_ind.view(dp_ind.size(0), -1).unsqueeze(2)
        dp_ind = dp_ind.expand(dp_ind.size(0), dp_ind.size(1), 3)
        verts = verts.gather(1, dp_ind).view(dp_ind.size(0), -1, 3, 3)
        dp_rat = dp_rat.unsqueeze(2).expand(dp_rat.size(0), dp_rat.size(1), 3, 3)
        verts = torch.sum(verts * dp_rat, 2)

        ## dp2d
        ind_ = ind[mask == 1].view(-1, 1)
        box_center = torch.cat((ind_ % opt.output_res, ind_ // opt.output_res), 1).type(torch.float32)

        pred = transpose_and_gather_feat(wh, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_wh = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(cd, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        box_cd = pred[mask_pre == 1].view(-1, 2)

        pred = transpose_and_gather_feat(camera, ind)
        mask_pre = mask.unsqueeze(2).expand_as(pred)
        camera = pred[mask_pre == 1].view(-1, 3)

        c = (box_center + box_cd + camera[:, 1:]) * opt.down_ratio
        f = (camera[:, 0].abs() * torch.sqrt(box_wh[:,0].abs() * box_wh[:,1].abs()) * opt.down_ratio * opt.camera_pose_z).view(-1,1) # TODO give camera off bias a initial value

        # kp3d = torch.matmul(kp3d, self.Rx) # global rotation
        verts[:,:, 2] = verts[:,:, 2] + opt.camera_pose_z # let z be positive

        verts = verts / torch.unsqueeze(verts[:,:,2], 2) # homogeneous vector
        dp2d = verts[:,:,:2] * f.view(-1,1,1) + c.view(-1,1,2) # camera transformation

        return dp2d


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