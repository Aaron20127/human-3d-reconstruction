
import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from dataset.dataloader import coco_data_loader, lsp_data_loader, hum36m_data_loader, multi_data_loader
from models.model import HmrNetBase, ModelWithLoss, HmrLoss
from utils.debugger import Debugger

from utils.util import AverageMeter, Clock, str_time, show_net_para


class HMRTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.loss_stats = ['loss', 'hm', 'wh', 'cd',
                            'pose', 'shape', 'kp2d']
        self.start_epoch = 0
        self.min_val_loss = 1e10  # for save best val model

        self.build_model(opt)
        self.create_data_loader(opt)


    def build_model(self, opt):
        print('start building model ...')

        ### 1.object detection model
        model = HmrNetBase()
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        if os.path.exists(opt.load_model):
            model, optimizer, self.start_epoch = \
              self.load_model(
                  model, opt.load_model, opt.device,
                  optimizer, opt.resume,
                  opt.lr, opt.lr_step
              )

        self.model = model
        self.optimizer = optimizer
        self.model_with_loss = \
            nn.DataParallel(ModelWithLoss(model, HmrLoss())).to(opt.device)

        show_net_para(model)
        print('finished build model.')


    def create_data_loader(self, opt):
        print('start creating data loader ...')

        loaders = []

        if opt.batch_size_coco > 0:
            loaders.append(coco_data_loader())
        if opt.batch_size_lsp > 0:
            loaders.append(lsp_data_loader())
        if opt.batch_size_hum36m > 0:
            loaders.append(hum36m_data_loader())

        if not loaders:
            assert 0, 'no data loaders.'

        self.train_loader = multi_data_loader(loaders)

        print('finished create data loader.')


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
        for iter_id in range(num_iters):
            batch = next(data_loader)
            if iter_id >= num_iters:
                break
            data_time.update(clock.elapsed())

            # forward
            for k in batch['label']:
                batch['label'][k] = batch['label'][k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch['label'])
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
                     loss_stats[l].mean().item(), batch['label']['input'].size(0))
                msg += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if not opt.hide_data_time:
                msg +=  '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print(msg)

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            del output, loss, loss_stats
            clock.elapsed()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = clock_ETA.total() / 60. # spending time of each epoch.
        return ret, results


    def train(self):
        print('start training ...')

        opt = self.opt
        start_epoch = self.start_epoch
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

            if opt.save_intervals > 0 and \
               epoch % opt.save_intervals == 0:
                self.save_model(os.path.join(opt.save_dir, 'model_epoch_{}.pth'.format(epoch)),
                          epoch, self.model, self.optimizer)

            self.save_model(os.path.join(opt.save_dir,'model_last.pth'),
                      epoch, self.model, self.optimizer)


    def val(self):
        _, preds = self.run_epoch('val', 0, self.val_loader)
        self.val_loader.dataset.run_eval(preds, self.opt.save_dir)


    def debug(self, batch, output, iter_id):
        opt = self.opt
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




    def load_model(self, model, model_path, device, optimizer=None, resume=False,
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

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
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