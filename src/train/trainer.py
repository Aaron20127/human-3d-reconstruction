
import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from dataset.dataloader import coco_data_loader, lsp_data_loader, hum36m_data_loader, multi_data_loader, val_coco_data_loader, val_hum36m_data_loader
from models.model import HmrNetBase, ModelWithLoss, HmrLoss
from utils.debugger import Debugger

from utils.util import AverageMeter, Clock, str_time, show_net_para, sigmoid
from utils.decode import decode
from utils.evaluate import covert_eval_data, eval


class HMRTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.loss_stats = ['loss', 'hm', 'wh', 'cd',
                            'pose', 'shape', 'kp2d', 'kp3d']
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
                  opt.lr
              )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=opt.lr_scheduler_factor, patience=opt.lr_scheduler_patience,
            verbose=True, threshold=opt.lr_scheduler_threshold, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-14)

        self.model = model
        self.optimizer = optimizer
        self.model_with_loss = \
            nn.DataParallel(ModelWithLoss(model, HmrLoss())).to(opt.device)

        show_net_para(model)
        print('finished build model.')


    def create_data_loader(self, opt):
        print('start creating data loader ...')

        ## train
        if not opt.val:
            loaders = []

            if opt.batch_size_coco > 0:
                loaders.append(coco_data_loader())
            if opt.batch_size_lsp > 0:
                loaders.append(lsp_data_loader())
            if opt.batch_size_smpl > 0:
                loaders.append(hum36m_data_loader())

            if not loaders:
                assert 0, 'no data loaders.'

            self.train_loader = multi_data_loader(loaders)

        ## val
        loaders = []
        if opt.val_batch_size_coco > 0:
            loaders.append(val_coco_data_loader())
        if opt.val_batch_size_smpl > 0:
            loaders.append(val_hum36m_data_loader())

        if len(loaders) == 0:
            self.val_loader = None
        else:
            self.val_loader = multi_data_loader(loaders)

        print('finished create data loader.')


    def write_log(self, phase, epoch, total_iters, num_iters, loss_states):
        logger = self.opt.logger
        text = time.strftime('%Y-%m-%d_%H-%M-%S: ')
        text += 'epoch:{:2}-{} |'.format(epoch, num_iters)
        for k, v in loss_states.items():
          logger.scalar_summary('{}_{}'.format(phase, k), v, total_iters)
          text += '{} {:8f} | '.format(k, v)
        logger.write(phase, text + '\n')


    def run_val(self, phase, epoch, total_iters=0, train_num_iters=0):
        opt = self.opt
        """ val """
        with torch.no_grad():
          loss_states = self.run_val_epoch(epoch, train_num_iters, self.val_loader)

        ## train
        if phase == 'train':
            ## save best model
            if loss_states['loss'] < self.min_val_loss:
                self.min_val_loss = loss_states['loss']
                self.save_model(os.path.join(self.opt.save_dir,
                              'model_best.pth'),
                               epoch, self.model)

            ## save log
            self.write_log('val', epoch, total_iters, train_num_iters, loss_states)


    def run_val_epoch(self, epoch, train_num_iters, data_loader):
        """ val """
        with torch.no_grad():
            model_with_loss = self.model_with_loss

            if len(self.opt.gpus_list) > 1:
                model_with_loss = self.model_with_loss.module # what this operation does?
            model_with_loss.eval()
            torch.cuda.empty_cache()

            ### 2. val
            opt = self.opt
            data_time, batch_time = AverageMeter(), AverageMeter()
            avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
            num_iters = len(data_loader)
            # num_iters = 10

            # get mAP
            eval_data = {
                'dts':[],
                'gts':[]
            }

            clock = Clock()
            clock_ETA = Clock()
            for iter_id in range(num_iters):
                batch = next(data_loader)
                data_time.update(clock.elapsed())

                # forward
                for k in batch['label']:
                    batch['label'][k] = batch['label'][k].to(device=opt.device, non_blocking=True)
                output, loss, loss_stats = model_with_loss(batch['label'])
                batch_time.update(clock.elapsed())

                # train message
                msg = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase='val',
                    total=str_time(clock_ETA.total()),
                    eta=str_time(clock_ETA.elapsed() * (num_iters - iter_id)))

                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['label']['input'].size(0))
                    msg += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not opt.hide_data_time:
                    msg += '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                           '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

                if opt.print_iter > 0:
                    if iter_id % opt.print_iter == 0:
                        print(msg)

                ## debug
                if opt.debug > 0:
                    self.debug(batch, output, iter_id)

                if opt.eval_average_precision:
                    covert_eval_data(output, batch, iter_id, eval_data, opt.eval_data_type, opt.score_thresh)

                del output, loss, loss_stats
                clock.elapsed()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = clock_ETA.total() / 60.  # spending time of each epoch.
        if opt.eval_average_precision:
            ret['mAP'] = eval(eval_data, opt.iou_thresh,
                              data_type=opt.eval_data_type,
                              save_path=opt.pr_curve_dir,
                              image_id='{}_{}'.format(epoch, train_num_iters))
        return ret


    def run_train_epoch(self, epoch):
        """ run one epoch """
        data_loader = self.train_loader
        model_with_loss = self.model_with_loss
        model_with_loss.train()

        ### 2. train
        opt = self.opt
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters

        clock = Clock()
        clock_ETA = Clock()
        for iter_id in range(num_iters):
            iter_id += 1

            batch = next(data_loader)
            if iter_id > num_iters:
                break
            data_time.update(clock.elapsed())

            # forward
            for k in batch['label']:
                batch['label'][k] = batch['label'][k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch['label'])

            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)
            batch_time.update(clock.elapsed())

            # train message
            msg = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase='train',
                total=str_time(clock_ETA.total()),
                eta=str_time(clock_ETA.elapsed() * (num_iters-iter_id)))

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                     loss_stats[l].mean().item(), batch['label']['input'].size(0))
                msg += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)


            if not opt.hide_data_time:
                msg +=  '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            ## mag
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print(msg)

            ## debug
            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            ## val
            if opt.val_iter_interval > 0 and \
                iter_id % opt.val_iter_interval == 0:
                    self.run_val('train', epoch, (epoch-1) *num_iters + iter_id, iter_id)

            ## log
            if opt.log_iters > 0 and \
                iter_id % opt.log_iters == 0:
                    ret = {k: v.avg for k, v in avg_loss_stats.items()}
                    ret['time'] = clock_ETA.total() / 60.
                    self.write_log('train', epoch, (epoch-1) *num_iters + iter_id, iter_id, ret)

            ## save model
            if opt.save_iter_interval > 0 and \
               iter_id % opt.save_iter_interval == 0:
                self.save_model(os.path.join(opt.save_dir, 'model_epoch_{}_{}.pth'.format(epoch, iter_id)),
                          epoch, self.model, self.optimizer)

            del output, loss, loss_stats
            clock.elapsed()

        if opt.log_iters > 0 and \
                iter_id % opt.log_iters != 0:
            ret = {k: v.avg for k, v in avg_loss_stats.items()}
            ret['time'] = clock_ETA.total() / 60.
            self.write_log('train', epoch, (epoch-1) *num_iters + iter_id, iter_id, ret)
        return num_iters


    def train(self):
        print('start training ...')

        opt = self.opt
        start_epoch = self.start_epoch

        for epoch in range(start_epoch+1, opt.num_epochs+1):
            total_iter = self.run_train_epoch(epoch)

            if opt.val_epoch_interval > 0 and \
                epoch % opt.val_epoch_interval == 0:
                if self.val_loader is not None:
                    self.run_val('train', epoch, total_iter)

            if opt.save_epoch_interval > 0 and \
               epoch % opt.save_epoch_interval == 0:
                self.save_model(os.path.join(opt.save_dir, 'model_epoch_{}.pth'.format(epoch)),
                          epoch, self.model, self.optimizer)

            self.save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                      epoch, self.model, self.optimizer)


    def val(self):
        if self.val_loader is not None:
            self.run_val('val', 0)


    def debug(self, batch, output, iter_id):
        opt = self.opt

        pred = decode(output, thresh=opt.score_thresh)

        debugger = Debugger(opt.smpl_path)

        for i, img in enumerate(batch['label']['input']):
            img = batch['label']['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img + 1) / 2 * 255.), 0, 255).astype(np.uint8)

            # gt heat map
            # gt_box_hm = debugger.gen_colormap(batch['label']['box_hm'][i].detach().cpu().numpy())
            # debugger.add_blend_img(img, gt_box_hm, 'gt_box_hm')

            # gt bbox, key points
            gt_id = 'gt_bbox_kp2d'
            debugger.add_img(img, img_id=gt_id)
            for b_gt in batch['gt']['gt']:
                for obj in b_gt:
                    debugger.add_bbox(obj['bbox'][0].detach().cpu().numpy(), img_id=gt_id)
                    debugger.add_kp2d(obj['kp2d'][0].detach().cpu().numpy(), img_id=gt_id)

            # pred heat map
            # pred_box_hm = debugger.gen_colormap(output['box_hm'][i].detach().cpu().numpy())
            # debugger.add_blend_img(img, pred_box_hm, 'pred_box_hm')

            # pred bbox, key points
            bbox_kp2d_id = 'pred_bbox_kp2d'
            smpl_id = 'pred_smpl'
            debugger.add_img(img, img_id=bbox_kp2d_id)
            debugger.add_img(img, img_id=smpl_id)

            obj = pred[i]
            if len(obj) > 0:
                for j in range(obj['bbox'].shape[0]):
                    debugger.add_bbox(obj['bbox'][j],conf=obj['score'][j], img_id=bbox_kp2d_id)
                    debugger.add_smpl_kp2d(obj['pose'][j], obj['shape'][j], obj['camera'][j],
                                            img_id=smpl_id, bbox_img_id=bbox_kp2d_id)

            if opt.debug == 1:
                debugger.show_all_imgs(pause=True)

            if opt.debug == 2:
                debugger.save_all_imgs(iter_id, opt.debug_image_dir)

            if opt.debug == 3:
                debugger.save_all_imgs(iter_id, opt.debug_image_dir)
                debugger.save_objs(iter_id, opt.debug_obj_dir)


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