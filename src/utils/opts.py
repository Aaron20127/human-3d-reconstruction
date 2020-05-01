from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np


parser = argparse.ArgumentParser()
agt = parser.add_argument


# basic experiment setting
agt('--eval', action='store_true', help='train or value.')
agt('--demo', default='', help='path to image/ image folders/ video. or webcam')
agt('--load_model', default='C:/Users/Lee/Desktop/model_last.pth', help='path to pretrained model')
agt('--debug', type=int, default=1, help='level of visualization.'
                                       '1: only show the final detection results'
                                       '2: show the network output features'
                                       '3: use matplot to display'  # useful when lunching training with ipython notebook
                                       '4: save all visualizations to disk')
agt('--resume', action='store_true', help='resume an experiment. '
                                        'Reloaded the optimizer parameter and '
                                        'set load_model to model_last.pth '
                                        'in the exp dir if load_model is empty.')

# system
agt('--gpus', default='0,1', help='-1 for CPU, use comma for multiple gpus')
agt('--num_workers', type=int, default=0, help='dataloader threads. 0 for single-thread.')
agt('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
agt('--seed', type=int, default=317, help='random seed')  # from CornerNet

# log
agt('--title', default='demo', help='title of log file.')
agt('--print_iter', type=int, default=1, help='disable progress bar and print to screen.')


# email
agt('--mail_host', default="smtp.163.com", help='SMTP server.')
agt('--mail_user', default='test_aaron@163.com', help='email name.')
agt('--mail_pwd', default='RNHKUQZIGAXXRIYZ', help='SMTP server password.')
agt('--mail_sender', default='test_aaron@163.com', help='mail sender.')
agt('--mail_receiver', default='lwalgorithm@163.com', help='mail receiver.')

# model
agt('--not_use_dcn', action='store_true', help='whether or not to use the DeformConv convolution')
agt('--down_ratio', type=int, default=4, help='output stride. Currently only supports 4.')


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
agt('--data_dir', default='/opt/ZHENGXU/DATASET/COCO2017/', help='not use the random crop data augmentation from CornerNet.')
agt('--not_rand_crop', action='store_true', help='not use the random crop data augmentation from CornerNet.')
agt('--shift', type=float, default=0.1, help='when not using random crop apply shift augmentation.')
agt('--scale', type=float, default=0.4, help='when not using random crop apply scale augmentation.')
agt('--rotate', type=float, default=0, help='when not using random crop apply rotation augmentation.')
agt('--flip', type=float, default=0.5, help='probability of applying flip augmentation.')
agt('--no_color_aug', action='store_true', help='not use the color augmenation from CornerNet')






# other parameters
opt = parser.parse_args()


# dataset path
opt.coco_data_set =  ['coco2014', 'coco2017']
opt.lsp_data_set =   ['lsp', 'lsp_ext']
opt.hum36_data_set = ['hum3.6m']

opt.train_adv_set = ['mosh']
opt.eval_set = ['up3d']


opt.data_set_path = {
    'coco2014': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2014',
    'coco2017': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017',
    'lsp': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp',
    'lsp_ext': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp_extend',
    # 'ai-ch':'E:/HMR/data/ai_challenger_keypoint_train_20170902',
    # 'mpi-inf-3dhp':'E:/HMR/data/mpi_inf_3dhp',
    'hum3.6m': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/hum36m-toy',
    'mosh': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/mosh/neutrMosh',
    'up3d': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/up-3d'
}


opt.gpus_str = opt.gpus
opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

opt.fix_res = not opt.keep_res
print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
opt.reg_offset = not opt.not_reg_offset
opt.reg_bbox = not opt.not_reg_bbox
opt.hm_hp = not opt.not_hm_hp
opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

if opt.head_conv == -1:  # init default head_conv
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
dir_name = opt.title.replace(" ", "_") + '_' + time.strftime('%Y-%m-%d_%H-%M-%S')
opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id, dir_name)
opt.debug_dir = os.path.join(opt.save_dir, 'debug')
print('The output will be saved to ', opt.save_dir)