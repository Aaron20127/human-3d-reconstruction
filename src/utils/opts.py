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
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

#
# # basic experiment setting
# agt('--eval', action='store_true', help='train or value.')
# agt('--demo', default='', help='path to image/ image folders/ video. or webcam')
# agt('--load_model', default='C:/Users/Lee/Desktop/model_last.pth', help='path to pretrained model')
# agt('--debug', type=int, default=1, help='level of visualization.'
#                                        '1: only show the final detection results'
#                                        '2: show the network output features'
#                                        '3: use matplot to display'  # useful when lunching training with ipython notebook
#                                        '4: save all visualizations to disk')
# agt('--resume', action='store_true', help='resume an experiment. '
#                                         'Reloaded the optimizer parameter and '
#                                         'set load_model to model_last.pth '
#                                         'in the exp dir if load_model is empty.')
#
# # system
# agt('--gpus', default='0,1', help='-1 for CPU, use comma for multiple gpus')
# agt('--num_workers', type=int, default=0, help='dataloader threads. 0 for single-thread.')
# agt('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
# agt('--seed', type=int, default=317, help='random seed')  # from CornerNet
#
# # log
# agt('--title', default='demo', help='title of log file.')
# agt('--print_iter', type=int, default=1, help='disable progress bar and print to screen.')
#
#
# # email
# agt('--mail_host', default="smtp.163.com", help='SMTP server.')
# agt('--mail_user', default='test_aaron@163.com', help='email name.')
# agt('--mail_pwd', default='RNHKUQZIGAXXRIYZ', help='SMTP server password.')
# agt('--mail_sender', default='test_aaron@163.com', help='mail sender.')
# agt('--mail_receiver', default='lwalgorithm@163.com', help='mail receiver.')
#
# # model
# agt('--not_use_dcn', action='store_true', help='whether or not to use the DeformConv convolution')
# agt('--down_ratio', type=int, default=4, help='output stride. Currently only supports 4.')
#
#
# # input
# agt('--input_res', type=int, default=-1, help='input height and width. -1 for default from '
#                                               'dataset. Will be overriden by input_h | input_w')
# agt('--input_h', type=int, default=-1, help='input height. -1 for default from dataset.')
# agt('--input_w', type=int, default=-1, help='input width. -1 for default from dataset.')
#
#
# # train
# agt('--lr', type=float, default=1.25e-4, help='learning rate for batch size 32.')
# agt('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')
# agt('--num_epochs', type=int, default=140, help='total training epochs.')
# agt('--batch_size', type=int, default=32, help='batch size')
# agt('--master_batch_size', type=int, default=-1, help='batch size on the master gpu.')
# agt('--num_iters', type=int, default=-1, help='default: #samples / batch_size.')
# agt('--val_intervals', type=int, default=5, help='number of epochs to run validation.')
# agt('--trainval', action='store_true', help='include validation in training and '
#                                           'test on test set')
#
# # test
# agt('--flip_test', action='store_true', help='flip data augmentation.')
# agt('--test_scales', type=str, default='1', help='multi scale test augmentation.')
# agt('--nms', action='store_true', help='run nms in testing.')
# agt('--K', type=int, default=100, help='max number of output objects.')
# agt('--not_prefetch_test', action='store_true', help='not use parallal data pre-processing.')
# agt('--fix_res', action='store_true', help='fix testing resolution or keep the original resolution')
# agt('--keep_res', action='store_true', help='keep the original resolution during validation.')
#
# # dataset
# agt('--data_dir', default='/opt/ZHENGXU/DATASET/COCO2017/', help='not use the random crop data augmentation from CornerNet.')
# agt('--not_rand_crop', action='store_true', help='not use the random crop data augmentation from CornerNet.')
# agt('--shift', type=float, default=0.1, help='when not using random crop apply shift augmentation.')
# agt('--scale', type=float, default=0.4, help='when not using random crop apply scale augmentation.')
# agt('--rotate', type=float, default=0, help='when not using random crop apply rotation augmentation.')
# agt('--flip', type=float, default=0.5, help='probability of applying flip augmentation.')
# agt('--no_color_aug', action='store_true', help='not use the color augmenation from CornerNet')
#
#




################# args #################
# email
agt('--mail_host', default="smtp.163.com", help='SMTP server.')
agt('--mail_user', default='test_aaron@163.com', help='email name.')
agt('--mail_pwd', default='RNHKUQZIGAXXRIYZ', help='SMTP server password.')
agt('--mail_sender', default='test_aaron@163.com', help='mail sender.')
agt('--mail_receiver', default='lwalgorithm@163.com', help='mail receiver.')

# log
agt('--exp_id', default='demo', help='experiments name.')
agt('--note', default='none', help='some notes for the experiment.')
agt('--print_iter', default=1, type=int, help='disable progress bar and print to screen.')
agt('--debug', default=0, type=int, help='level of visualization.')
agt('--hide_data_time', action='store_true', help='hide print of time of model and dataload.')

# system
agt('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
agt('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
agt('--seed', default=317, type=int, help='random seed')

# network
agt('--use_dcn', action='store_true', help='whether or not to use the DeformConv convolution')

# loss
agt('--hm_weight', type=float, default=1, help='loss weight for keypoint heatmaps.')
agt('--wh_weight', type=float, default=1, help='loss weight for bounding box width and height.')
agt('--cd_weight', type=float, default=1, help='loss weight for bounding box center decimal.')
agt('--pose_weight', type=float, default=1, help='loss weight for keypoint heatmaps.')
agt('--shape_weight', type=float, default=1, help='loss weight for bounding box width and height.')
agt('--kp2d_weight', type=float, default=1, help='loss weight for bounding box center decimal.')

# train
agt('--val', action='store_true', help='train or eval.')
agt('--pre_trained_model', default='', help='Pretraining model')
agt('--resume', action='store_true', help='resume an experiment.')
agt('--val_intervals', default=-1, type=int,  help='number of epochs to run validation.')
agt('--num_iters', default=-1, type=int, help='default: #samples / batch_size.')
agt('--num_epochs', default=1, type=int, help='.')
agt('--lr', default=1.25e-4, type=float,  help='learning rate for batch size 32.')
agt('--lr_step', default='90,120', type=str, help='drop learning rate by 10.')

# dataset
agt('--batch_size_coco', default=0, type=int,  help='0: donot use this data set.')
agt('--batch_size_lsp',  default=2, type=int, help='0: donot use this data set.')
agt('--batch_size_hum36m', default=0, type=int,  help='0: donot use this data set.')
agt('--num_workers', default=0, type=int, help='dataloader threads. 0 for single-thread.')


opt = parser.parse_args()



################ network ################
opt.heads = {'box_hm':1, 'box_wh':2, 'box_cd':2, 'pose':72, 'shape':10, 'camera':3}
opt.smpl_path = os.path.join(root_dir, 'data', 'neutral_smpl_with_cocoplus_reg.pkl')



################ dataset ################
# opt.train_set =  ['coco2014', 'coco2017', 'lsp', 'lsp_ext', 'hum3.6m']
opt.coco_data_set=['coco2017']
opt.lsp_data_set=['lsp']
opt.hum36m_data_set=['hum36m']

opt.train_set =  ['coco2017', 'lsp', 'hum36m']
opt.train_adv_set = ['mosh']
opt.eval_set = ['up3d']


opt.data_set_path = {
    'coco2014': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2014',
    'coco2017': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017',
    'lsp': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp/',
    'lsp_ext': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp_extend',
    # 'ai-ch':'E:/HMR/data/ai_challenger_keypoint_train_20170902',
    # 'mpi-inf-3dhp':'E:/HMR/data/mpi_inf_3dhp',
    'hum36m': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/hum36m-toy',
    'mosh': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/mosh/neutrMosh',
    'up3d': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/up-3d'
}


################### preprocess ##################
"""log"""
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
exp_dir = os.path.join(root_dir, 'exp')
dir_name = opt.exp_id.replace(" ", "_")
opt.save_dir = os.path.join(exp_dir, dir_name)
opt.debug_dir = os.path.join(opt.save_dir, 'debug')

"""train"""
opt.lr_step = [int(i) for i in opt.lr_step.split(',')]