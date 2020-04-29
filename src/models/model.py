import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from torch.nn import nn

from losses import FocalLoss, RegL1Loss
from model_util import batch_orth_proj
from utils.opts import opt
from network.dla from DlaSeg
from network.smpl from SMPL


class HmrLoss(nn.Module):
    def __init__(self, opt):
        super(HmrLoss, self).__init__()
        self.opt = opt
        self.crit_hm = FocalLoss()
        self.crit_reg = RegL1Loss()

    def forward(self, output, batch):
        """
            batch
            {
                'hm':         '(n,1,128,128)',

                'wh':         '(n, max_obj, 2)',
                'wh_ind':     '(n, max_obj)',
                'wh_mask':    '(n, max_obj)',

                'kp_2d':      '(n, 14, 3)', # 第三列是否可见可以作为索引
                'kp_3d':      '(n, 19, 3)',

                'cam':        '(n, 3)',
                'shape':      '(n, 10)',
                'pose':       '(n, 72)',
            }

            output
            {
                'hm':           '(n,1,128,128)',
                'wh':           '(n,2,128,128)',
                'pose':         '(n,72,128,128)',
                'shape':        '(n,10,128,128)',
                'cam':          '(n, 3,128,128)',
            }
        """
        hm_loss, wh_loss = 0, 0
        pose_loss, shape_loss = 0, 0
        2d_loss, 3d_loss = 0, 0

        ## 1.loss of object bbox
        # heat map loss of objects center
        output['hm'] = _sigmoid(output['hm']) # do sigmoid
        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        # height and width of bbox, use l1 loss and do not use heat map
        wh_loss = self.crit_reg(output['wh'], batch['wh_mask'],
                                 batch['wh_ind'], batch['wh'])

        ## 2.loss of humman key points


        ##
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
               opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats


class HmrNetBase(nn.Module):
    def __init__(self):
        super(HmrNetBase, self).__init__()

        print('start creating sub modules...')
        self._create_sub_modules()


    def _create_sub_modules(self):
        self.encoder = DlaSeg(opt.heads,
                              not_use_dcn=opt.not_use_dcn)
        print('finished create the encoder module ...')

        self.smpl = SMPL(opt.smpl_model_path)
        print('finished create the smpl module ...')


    def forward(self, input):
        ## encoder
        ret = self.encoder(input)

        ## smpl
        b, w, h, _ = ret['cam'].size()
        cam  = ret['cam'].view(-1,3) # (batch*128*128, 3)
        pose = ret['pose'].view(-1,75)
        shape = ret['shape'].view(-1,10)

        j3d = self.smpl(beta=shape, theta=pose)
        j2d = batch_orth_proj(j3d, cam)

        j3d = j3d.view(b ,w, h, j3d.shape[-2], j3d.shape[-1])
        j2d = j3d.view(b ,w, h, j2d.shape[-2], j2d.shape[-1])

        ret.update({'kp_2d': j2d, 'kp_3d':j3d})

        return ret