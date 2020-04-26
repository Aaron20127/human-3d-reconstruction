from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    agt = self.parser.add_argument

    # basic experiment setting
    agt('task', default='ctdet', help='ctdet | ddd | multi_pose | exdet')
    agt('--dataset', default='coco', help='coco | kitti | coco_hp | pascal')
    agt('--test', action='store_true', help='train or evalue.')
    agt('--demo', default='', help='path to image/ image folders/ video. or webcam')
    agt('--load_model', default='C:/Users/Lee/Desktop/model_last.pth', help='path to pretrained model')
    agt('--exp_id', default='default')
    agt('--debug', type=int, default=1, help='level of visualization.'
                                             '1: only show the final detection results'
                                             '2: show the network output features'
                                             '3: use matplot to display' # useful when lunching training with ipython notebook
                                             '4: save all visualizations to disk')
    agt('--resume', action='store_true', help='resume an experiment. '
                                              'Reloaded the optimizer parameter and '
                                              'set load_model to model_last.pth '
                                              'in the exp dir if load_model is empty.')

    # system
    agt('--gpus', default='0,1',  help='-1 for CPU, use comma for multiple gpus')
    agt('--num_workers', type=int, default=0, help='dataloader threads. 0 for single-thread.')
    agt('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
    agt('--seed', type=int, default=317, help='random seed') # from CornerNet


    # log
    agt('--title', default='demo', help='title of log file.')
    agt('--print_iter', type=int, default=1, help='disable progress bar and print to screen.')
    agt('--hide_data_time', action='store_true', help='not display time during training.')
    agt('--save_all', action='store_true', help='save model to disk every 5 epochs.')
    agt('--metric', default='loss', help='main metric to save best model')
    agt('--vis_thresh', type=float, default=0.3, help='visualization threshold.')
    agt('--debugger_theme', default='white', choices=['white', 'black'])


    # email
    agt('--mail_host',     default="smtp.163.com",         help='SMTP server.')
    agt('--mail_user',     default='lwalgorithm@163.com',  help='email name.')
    agt('--mail_pwd',      default='RZVYBFVFKEASVHQO',     help='SMTP server password.')
    agt('--mail_sender',   default='lwalgorithm@163.com',  help='mail sender.')
    agt('--mail_receiver', default='lwalgorithm@163.com',  help='mail receiver.')

    
    # model
    agt('--not_use_dcn',  action='store_true',  help='whether or not to use the DeformConv convolution')
    agt('--down_ratio', type=int, default=4, help='output stride. Currently only supports 4.')
    agt('--arch', default='dla_34', help='model architecture. Currently tested'
                                         'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                         'dlav0_34 | dla_34 | hourglass')
    agt('--head_conv', type=int, default=-1, help='conv layer channels for output head'
                                                  '0 for no conv layer'
                                                  '-1 for default setting: '
                                                  '64 for resnets and 256 for dla.')


    # input
    agt('--input_res', type=int, default=-1, help='input height and width. -1 for default from '
                                                  'dataset. Will be overriden by input_h | input_w')
    agt('--input_h', type=int, default=-1, help='input height. -1 for default from dataset.')
    agt('--input_w', type=int, default=-1, help='input width. -1 for default from dataset.')


    # train
    agt('--lr', type=float, default=1.25e-4, help='learning rate for batch size 32.')
    agt('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')
    agt('--num_epochs', type=int, default=140, help='total training epochs.')
    agt('--batch_size', type=int, default=32, help='batch size')
    agt('--master_batch_size', type=int, default=-1, help='batch size on the master gpu.')
    agt('--num_iters', type=int, default=-1, help='default: #samples / batch_size.')
    agt('--val_intervals', type=int, default=5, help='number of epochs to run validation.')
    agt('--trainval', action='store_true', help='include validation in training and '
                                                'test on test set')

    # test
    agt('--flip_test', action='store_true', help='flip data augmentation.')
    agt('--test_scales', type=str, default='1', help='multi scale test augmentation.')
    agt('--nms', action='store_true', help='run nms in testing.')
    agt('--K', type=int, default=100, help='max number of output objects.')
    agt('--not_prefetch_test', action='store_true', help='not use parallal data pre-processing.')
    agt('--fix_res', action='store_true', help='fix testing resolution or keep the original resolution')
    agt('--keep_res', action='store_true', help='keep the original resolution during validation.')


    # dataset
    agt('--data_dir', default='D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017',
                      help='not use the random crop data augmentation from CornerNet.')
    agt('--not_rand_crop', action='store_true', help='not use the random crop data augmentation from CornerNet.')
    agt('--shift', type=float, default=0.1, help='when not using random crop apply shift augmentation.')
    agt('--scale', type=float, default=0.4, help='when not using random crop apply scale augmentation.')
    agt('--rotate', type=float, default=0, help='when not using random crop apply rotation augmentation.')
    agt('--flip', type = float, default=0.5, help='probability of applying flip augmentation.')
    agt('--no_color_aug', action='store_true', help='not use the color augmenation from CornerNet')


    # ddd
    agt('--aug_ddd', type=float, default=0.5, help='probability of applying crop augmentation.')
    agt('--kitti_split', default='3dop', help='different validation split for kitti: 3dop | subcnn')
    agt('--rect_mask', action='store_true', help='for ignored object, apply mask on the '
                                                 'rectangular region or just center point.')

    # loss
    agt('--mse_loss', action='store_true', help='use mse loss or focal loss to train keypoint heatmaps.')


    # ctdet
    agt('--reg_loss', default='l1', help='regression loss: sl1 | l1 | l2')
    agt('--hm_weight', type=float, default=1, help='loss weight for keypoint heatmaps.')
    agt('--off_weight', type=float, default=1, help='loss weight for keypoint local offsets.')
    agt('--wh_weight', type=float, default=0.1, help='loss weight for bounding box size.')


    # multi_pose
    agt('--aug_rot', type=float, default=0, help='probability of applying rotation augmentation.')
    agt('--hp_weight', type=float, default=1, help='loss weight for human pose offset.')
    agt('--hm_hp_weight', type=float, default=1, help='loss weight for human keypoint heatmap.')


    # ddd
    agt('--dep_weight', type=float, default=1, help='loss weight for depth.')
    agt('--dim_weight', type=float, default=1, help='loss weight for 3d bounding box size.')
    agt('--rot_weight', type=float, default=1, help='loss weight for orientation.')
    agt('--peak_thresh', type=float, default=0.2)


    # task
    # ctdet
    agt('--norm_wh', action='store_true', help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    agt('--cat_spec_wh', action='store_true', help='category specific bounding box size.')
    agt('--not_reg_offset', action='store_true', help='not regress local offset.')
    agt('--dense_wh', action='store_true', help='apply weighted regression near center or '
                                                'just apply regression on center point.')


    # exdet
    agt('--agnostic_ex', action='store_true',  help='use category agnostic extreme points.')
    agt('--scores_thresh', type=float, default=0.1, help='threshold for extreme point heatmap.')
    agt('--center_thresh', type=float, default=0.3, help='threshold for centermap.')
    agt('--aggr_weight', type=float, default=0.0, help='edge aggregation weight.')


    # multi_pose
    agt('--dense_hp', action='store_true', help='apply weighted pose regression near center '
                                                'or just apply regression on center point.')
    agt('--not_hm_hp', action='store_true', help='not estimate human joint heatmap, '
                                                 'directly use the joint offset from center.')
    agt('--not_reg_hp_offset', action='store_true', help='not regress local offset for '
                                                         'human joint heatmaps.')
    agt('--not_reg_bbox', action='store_true', help='not regression bounding box size.')



  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp')
    # opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id, time.strftime('%Y-%m-%d_%H-%M-%S'))
    dir_name =  opt.title.replace(" ", "_") + '_' + time.strftime('%Y-%m-%d_%H-%M-%S')
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id, dir_name)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')

    self.update_dataset_info_and_set_heads(opt)
    return opt

  def update_dataset_info_and_set_heads(self, opt):
      input_h, input_w = 512, 512
      opt.mean = np.array([0.40789654, 0.44719302, 0.47026115],dtype=np.float32).reshape(1, 1, 3)
      opt.std = np.array([0.28863828, 0.27408164, 0.27809835],dtype=np.float32).reshape(1, 1, 3)

      # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
      input_h = opt.input_res if opt.input_res > 0 else input_h
      input_w = opt.input_res if opt.input_res > 0 else input_w
      opt.input_h = opt.input_h if opt.input_h > 0 else input_h
      opt.input_w = opt.input_w if opt.input_w > 0 else input_w
      opt.output_h = opt.input_h // opt.down_ratio
      opt.output_w = opt.input_w // opt.down_ratio
      opt.input_res = max(opt.input_h, opt.input_w)
      opt.output_res = max(opt.output_h, opt.output_w)


      if opt.task == 'ctdet':
          # assert opt.dataset in ['pascal', 'coco']
          opt.num_classes = 80
          opt.heads = {'hm': opt.num_classes,
                       'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
          if opt.reg_offset:
              opt.heads.update({'reg': 2})
      elif opt.task == 'multi_pose':
          # assert opt.dataset in ['coco_hp']
          opt.num_classes = 1
          opt.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],[11, 12], [13, 14], [15, 16]]
          opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
          if opt.reg_offset:
              opt.heads.update({'reg': 2})
          if opt.hm_hp:
              opt.heads.update({'hm_hp': 17})
          if opt.reg_hp_offset:
              opt.heads.update({'hp_offset': 2})
      else:
          assert 0, 'task not defined!'

      print('heads', opt.heads)

  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]},
      'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info['ctdet'])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
