from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys

from models.model import dla_net

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode, multi_pose_decode
from models.utils import _sigmoid

from utils.debugger import Debugger
from utils.post_process import ctdet_post_process, multi_pose_post_process


from utils.utils import AverageMeter
from .util import Clock, str_time, show_net_para

from datasets.coco import COCO
from datasets.coco_hp import COCOHP

class loss_obj_detection(torch.nn.Module):
  def __init__(self, opt):
    super(loss_obj_detection, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
      opt = self.opt
      hm_loss, wh_loss, off_loss = 0, 0, 0
      for s in range(opt.num_stacks):
          output = outputs[s]
          if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

          hm_loss += self.crit(output['hm'], batch['hm']) # heat map loss
          if opt.wh_weight > 0:
                if opt.dense_wh:
                      mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                      wh_loss += (
                        self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                        batch['dense_wh'] * batch['dense_wh_mask']) /
                        mask_weight)
                elif opt.cat_spec_wh:
                      wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh'])
                else:
                      wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh'])

          if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                     batch['ind'], batch['reg'])

      loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
             opt.off_weight * off_loss
      loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss}
      return loss, loss_stats

class loss_multi_pose(torch.nn.Module):
    def __init__(self, opt):
        super(loss_multi_pose, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        output = outputs[0]
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0

        ## 1.loss of object center
        output['hm'] = _sigmoid(output['hm']) # do sigmoid
        hm_loss += self.crit(output['hm'], batch['hm'])


        if opt.dense_hp:
            mask_weight = batch['dense_hps_mask'].sum() + 1e-4
            hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                     batch['dense_hps'] * batch['dense_hps_mask']) /
                        mask_weight)
        else:
            ## vector of object center to key points, use l1 loss and do not use heat map.
            hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                    batch['ind'], batch['hps'])

        if opt.wh_weight > 0:
            ## height and width of bbox, use l1 loss and do not use heat map
            wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                     batch['ind'], batch['wh'])

        if opt.reg_offset and opt.off_weight > 0:
            ## decimal of bbox center points, use l1 loss and do not use heat map
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                      batch['ind'], batch['reg'])

        ## 2.loss of humman key points
        ## decimal of humman key points center, use l1 loss and do not use heat map
        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss += self.crit_reg(
                output['hp_offset'], batch['hp_mask'],
                batch['hp_ind'], batch['hp_offset'])

        ## heat map of all key points (17,128,128)
        if opt.hm_hp and opt.hm_hp_weight > 0:
            if opt.hm_hp and not opt.mse_loss:
                output['hm_hp'] = _sigmoid(output['hm_hp'])
            hm_hp_loss += self.crit_hm_hp(output['hm_hp'], batch['hm_hp'])

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
               opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class HMRModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(HMRModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class HMRTrainer(object):
  def __init__(self, opt):
      self.opt = opt
      self.min_val_loss = 1e10  # for save best val model

      self._build_model(opt)
      self._create_data_loader(opt)

  def _build_model(self, opt):
      print('Starting building model.')

      ### 1.object detection model
      print('Building object detection model.')
      model = dla_net(opt.heads, not_use_dcn=opt.not_use_dcn)
      optimizer = torch.optim.Adam(model.parameters(), opt.lr)
      if os.path.exists(opt.load_model):
          model, optimizer, start_epoch = self.load_model(
              model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

      # self.model_obj_detection = nn.DataParallel(model).cuda()
      # self.loss_obj_detection = nn.DataParallel(loss_obj_detection()).cuda()

      self.model = model
      self.loss_stats, self.loss = self._get_losses(opt)
      self.model_with_loss = nn.DataParallel(HMRModelWithLoss(model, self.loss)).cuda()
      self.optimizer = optimizer

      show_net_para(model)
      print('Finished build model.')

  def _create_data_loader(self, opt):
      print('Create data loader.')

      dataset = COCO if opt.task == 'ctdet' else COCOHP

      self.val_loader = torch.utils.data.DataLoader(
          dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

      if not opt.test:
          self.train_loader = torch.utils.data.DataLoader(
              dataset(opt, 'train'), batch_size=opt.batch_size,
              shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)


  def run_train(self, epoch):
      """ train """
      ret, _ = self.run_epoch('train', epoch, self.train_loader)

      ## save train.txt
      logger = self.opt.logger
      text = time.strftime('%Y-%m-%d_%H-%M-%S: ')
      text += 'epoch: {} |'.format(epoch)
      for k, v in ret.items():
          logger.scalar_summary('train_{}'.format(k), v, epoch)
          text += '{} {:8f} | '.format(k, v)
      logger.write('train', text + '\n')

  def run_val(self, epoch):
      """ val """
      with torch.no_grad():
          ret, _ = self.run_epoch('val', epoch, self.val_loader)

      ## save val.txt
      logger = self.opt.logger
      text = time.strftime('%Y-%m-%d_%H-%M-%S: ')
      text += 'epoch: {} |'.format(epoch)
      for k, v in ret.items():
          logger.scalar_summary('val_{}'.format(k), v, epoch)
          text += '{} {:8f} | '.format(k, v)
      logger.write('val', text + '\n')

      ## save best model
      if ret['loss'] < self.min_val_loss:
          self.min_val_loss = ret['loss']
          self.save_model(os.path.join(self.opt.save_dir,
                          'model_best.pth'),
                           epoch, self.model)


  def run_epoch(self, phase, epoch, data_loader):
      """ run one epoch """
      model_with_loss = self.model_with_loss
      ### 1. train or eval
      if phase == 'train':
          model_with_loss.train()
      else:
          if len(self.opt.gpus) > 1:
              model_with_loss = self.model_with_loss.module
          model_with_loss.eval()
          torch.cuda.empty_cache()

      ### 2. train
      opt = self.opt
      results = {}
      data_time, batch_time = AverageMeter(), AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
      num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
      msg = ''

      clock = Clock()
      clock_ETA = Clock()
      for iter_id, batch in enumerate(data_loader):
          if iter_id >= num_iters:
              break
          data_time.update(clock.elapsed())

          # forward
          for k in batch:
              if k != 'meta':
                  batch[k] = batch[k].to(device=opt.device, non_blocking=True)
          output, loss, loss_stats = model_with_loss(batch)
          loss = loss.mean()
          if phase == 'train':
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
          batch_time.update(clock.elapsed())

          # training message
          msg = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
              epoch, iter_id, num_iters, phase=phase,
              total=str_time(clock_ETA.total()),
              eta=str_time(clock_ETA.elapsed() * (num_iters-iter_id)))

          for l in avg_loss_stats:
              avg_loss_stats[l].update(
                  loss_stats[l].mean().item(), batch['input'].size(0))
              msg += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

          if not opt.hide_data_time:
              msg +=  '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

          if opt.print_iter > 0:
              if iter_id % opt.print_iter == 0:
                  print(msg)

          if opt.debug > 0:
              self.debug(batch, output, iter_id)

          if opt.test:
              self.save_result(output, batch, results)

          del output, loss, loss_stats
          clock.elapsed()


      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      ret['time'] = clock_ETA.total() / 60. # spending time of each epoch.
      return ret, results


  def train(self):
      print('Starting training ...')

      opt = self.opt
      start_epoch = 0
      for epoch in range(start_epoch + 1, opt.num_epochs + 1):
          self.run_train(epoch)

          if opt.val_intervals > 0 and \
              epoch % opt.val_intervals == 0:
              self.run_val(epoch)

          if epoch in opt.lr_step:
              lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
              print('Drop LR to', lr)
              for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr

          self.save_model(os.path.join(opt.save_dir,'model_last.pth'),
                      epoch, self.model, self.optimizer)


  def val(self):
      _, preds = self.run_epoch('val', 0, self.val_loader)
      self.val_loader.dataset.run_eval(preds, self.opt.save_dir)


  def _get_losses(self, opt):
      if opt.task == 'ctdet':
          loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
          loss = loss_obj_detection(opt)
      elif opt.task == 'multi_pose':
          loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss',
                         'hp_offset_loss', 'wh_loss', 'off_loss']
          loss = loss_multi_pose(opt)
      else:
          assert 0, 'task not defined!'

      return loss_states, loss


  def debug(self, batch, output, iter_id):
      opt = self.opt

      if opt.task == 'ctdet':
          reg = output['reg'] if opt.reg_offset else None
          dets = ctdet_decode(
                output['hm'], output['wh'], reg=reg,
                cat_spec_wh=opt.cat_spec_wh, K=opt.K)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
          dets[:, :, :4] *= opt.down_ratio
          dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
          dets_gt[:, :, :4] *= opt.down_ratio
          for i in range(1):
                debugger = Debugger(
                    dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
                img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
                img = np.clip(((
                    img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
                pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hm')
                debugger.add_blend_img(img, gt, 'gt_hm')
                debugger.add_img(img, img_id='out_pred')
                for k in range(len(dets[i])):
                    if dets[i, k, 4] > opt.center_thresh:
                        debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

                debugger.add_img(img, img_id='out_gt')
                for k in range(len(dets_gt[i])):
                    if dets_gt[i, k, 4] > opt.center_thresh:
                        debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

                if opt.debug == 4:
                    debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
                else:
                    debugger.show_all_imgs(pause=True)

      elif opt.task == 'multi_pose':
          reg = output['reg'] if opt.reg_offset else None
          hm_hp = output['hm_hp'] if opt.hm_hp else None
          hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
          dets = multi_pose_decode(
              output['hm'], output['wh'], output['hps'],
              reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

          dets[:, :, :4] *= opt.input_res / opt.output_res
          dets[:, :, 5:39] *= opt.input_res / opt.output_res
          dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
          dets_gt[:, :, :4] *= opt.input_res / opt.output_res
          dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res

          for i in range(1):
              debugger = Debugger(
                  dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
              img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
              img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
              pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
              gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
              debugger.add_blend_img(img, pred, 'pred_hm')
              debugger.add_blend_img(img, gt, 'gt_hm')

              # out_pred_id = 'out_pred {}'.format(iter_id)
              out_pred_id = 'out_pred'
              debugger.add_img(img, img_id=out_pred_id)
              for k in range(len(dets[i])):
                  if dets[i, k, 4] > opt.center_thresh:
                      debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                             dets[i, k, 4], img_id=out_pred_id)
                      debugger.add_coco_hp(dets[i, k, 5:39], img_id=out_pred_id)

              # out_gt_id = 'out_gt {}'.format(iter_id)
              out_gt_id = 'out_gt'
              debugger.add_img(img, img_id=out_gt_id)
              for k in range(len(dets_gt[i])):
                  if dets_gt[i, k, 4] > opt.center_thresh:
                      debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                             dets_gt[i, k, 4], img_id=out_gt_id)
                      debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id=out_gt_id)

              if opt.hm_hp:
                  pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
                  gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
                  debugger.add_blend_img(img, pred, 'pred_hmhp')
                  debugger.add_blend_img(img, gt, 'gt_hmhp')

              if opt.debug == 4:
                  debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
              else:
                  debugger.show_all_imgs(pause=True)

      else:
          assert 0, 'task not defined!'


  def save_result(self, output, batch, results):
      opt = self.opt
      if opt.task == 'ctdet':
          reg = output['reg'] if self.opt.reg_offset else None
          dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
          dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
          results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

      elif opt.task == 'multi_pose':
          reg = output['reg'] if self.opt.reg_offset else None
          hm_hp = output['hm_hp'] if self.opt.hm_hp else None
          hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
          dets = multi_pose_decode(
              output['hm'], output['wh'], output['hps'],
              reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

          dets_out = multi_pose_post_process(
              dets.copy(), batch['meta']['c'].cpu().numpy(),
              batch['meta']['s'].cpu().numpy(),
              output['hm'].shape[2], output['hm'].shape[3])
          results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

      else:
          assert 0, 'task not defined!'


  def load_model(self, model, model_path, optimizer=None, resume=False,
                 lr=None, lr_step=None):
      start_epoch = 0
      checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
      print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
      state_dict_ = checkpoint['state_dict']
      state_dict = {}

      # convert data_parallal to model
      for k in state_dict_:
          if k.startswith('module') and not k.startswith('module_list'):
              state_dict[k[7:]] = state_dict_[k]
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

      # resume optimizer parameters
      if optimizer is not None and resume:
          if 'optimizer' in checkpoint:
              optimizer.load_state_dict(checkpoint['optimizer'])
              start_epoch = checkpoint['epoch']
              start_lr = lr
              for step in lr_step:
                  if start_epoch >= step:
                      start_lr *= 0.1
              for param_group in optimizer.param_groups:
                  param_group['lr'] = start_lr
              print('Resumed optimizer with start lr', start_lr)
          else:
              print('No optimizer parameters in checkpoint.')
      if optimizer is not None:
          return model, optimizer, start_epoch
      else:
          return model

  def save_model(self, path, epoch, model, optimizer=None):
      if isinstance(model, torch.nn.DataParallel):
          state_dict = model.module.state_dict()
      else:
          state_dict = model.state_dict()
      data = {'epoch': epoch,
              'state_dict': state_dict}
      if not (optimizer is None):
          data['optimizer'] = optimizer.state_dict()
      torch.save(data, path)