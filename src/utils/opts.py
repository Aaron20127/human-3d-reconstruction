from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np

from .util import pre_process

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
agt('--task', default='task', help='experiments name.')
agt('--subtask', default='demo', help='subtask of experiments name.')
agt('--note', default='none', help='some notes for the experiment.')
agt('--print_iter', default=1, type=int, help='disable progress bar and print to screen.')
agt('--debug', default=0, type=int, help='0 - donnot debug.'
                                         '1 - show image.'
                                         '2 - save image.'
                                         '3 - save image and obj.')
agt('--hide_data_time', action='store_true', help='hide print of time of model and dataload.')

# system
agt('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
agt('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
agt('--seed', default=317, type=int, help='random seed')
agt('--data_aug_seed', default=123, type=int, help='random seed for data aurgment')

# network
agt('--use_dcn', action='store_true', help='whether or not to use the DeformConv convolution')

# decode
agt('--score_thresh', default=0.3, type=float, help='be considered to be the lower limit of the score of an object')

# loss
agt('--hm_weight', default=1, type=float, help='loss weight for keypoint heatmaps.')
agt('--wh_weight', default=1, type=float, help='loss weight for bounding box width and height.')
agt('--cd_weight', default=1, type=float, help='loss weight for bounding box center decimal.')
agt('--pose_weight',  default=1, type=float, help='loss weight for keypoint heatmaps.')
agt('--shape_weight', default=1, type=float, help='loss weight for bounding box width and height.')
agt('--kp2d_weight',  default=1, type=float, help='loss weight for bounding box center decimal.')
agt('--kp3d_weight',  default=1, type=float, help='loss weight for bounding box center decimal.')
agt('--dp2d_weight',  default=1, type=float, help='loss weight for densepose points.')

agt('--pose_loss_type',  default=1, type=int, help='1 - rotating vector.'
                                                   '2 - euler angle.')
# agt('--kp2d_every_weight_train',  default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1',  help='weight for every 2d point.')
# agt('--kp2d_every_weight_val',  default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1',  help='weight for every 2d point.')
agt('--kp2d_every_weight_train',  default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1',  help='weight for every 2d point.')
agt('--kp2d_every_weight_val',  default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1',  help='weight for every 2d point.')


# train
agt('--smpl_type', default='basic', help='basic, cocoplus')
agt('--master_batch_size', default=-1, type=int, help='batch size for master GPU')
agt('--val', action='store_true', help='train or eval.')
agt('--lr', default=1.25e-4, type=float,  help='learning rate for batch size 32.')
agt('--lr_scheduler_factor', default=0.978, type=float,  help='learning factor.')
agt('--lr_scheduler_patience', default=200, type=float,  help='learning patience.')
agt('--lr_scheduler_threshold', default=0.01, type=float,  help='learning threshold.')

agt('--num_iters', default=-1, type=int, help='default: #samples / batch_size.')
agt('--num_epochs', default=10, type=int, help='.')
agt('--log_iters', default=10, type=int,  help='number of iters to log.')

agt('--load_model', default='', help='pretraining model')
agt('--resume', action='store_true', help='resume optimizer.')
agt('--save_iter_interval', default=4000, type=int,  help='number of epochs to save model.')
agt('--save_epoch_interval', default=1, type=int,  help='number of epochs to save model.')

agt('--not_shuffle_data_train', action='store_true', help='shuffle data loader of train.')

agt('--camera_pose_z', default=10, type=int, help='parameter z of camera pose of translation')


# value
agt('--val_num_iters', default=-1, type=int, help='default: #samples / batch_size.')
agt('--val_iter_interval', default=4000, type=int,  help='number of iter of one epoch to run validation.')
agt('--val_epoch_interval', default=1, type=int,  help='number of iter of one epoch to run validation.')
agt('--val_batch_size_coco', default=1, type=int,  help='0: donot use this data set.')
agt('--val_batch_size_hum36m', default=0, type=int,  help='0: donot use this data set.')
agt('--val_batch_size_3dpw', default=0, type=int,  help='0: donot use this data set.')

agt('--eval_average_precision', action='store_true', help='0: donot use this data set.')
agt('--iou_thresh', default='0.1,0.2,0.3', help='0: donot use this data set.')
agt('--eval_data_type', default='kps',  help='if , kps iou_thresh mean the reciprocal of mean joints loss.'
                                             'if , bbox iou_thresh mean the iou of bbox.')

agt('--val_shuffle_data', action='store_true', help='shuffle data loader of val.')
agt('--val_scale_data', default='1.0,1.01', help='scale data loader of val.')


# dataset
agt('--batch_size_coco', default=1, type=int,  help='0: donot use this data set.')
agt('--batch_size_lsp',  default=0, type=int, help='0: donot use this data set.')
agt('--batch_size_mpii',  default=0, type=int, help='0: donot use this data set.')
agt('--batch_size_hum36m', default=0, type=int,  help='0: donot use this data set.')
agt('--batch_size_3dpw', default=0, type=int,  help='0: donot use this data set.')
agt('--num_workers', default=0, type=int, help='dataloader threads. 0 for single-thread.')

agt('--min_bbox_area', default=256, type=float, help='minimum number of oringinal visible points of kp2d to train.')
agt('--min_vis_kps', default=6, type=int, help='minimum number of oringinal visible points of kp2d to train.')
agt('--keep_truncation_kps', action='store_true', help='keep points of kp2d out of image when trunction.')
agt('--min_truncation_kps', default=10, type=int, help='total key points of trunction people.')
agt('--min_truncation_kps_in_image', default=6, type=int, help='minimum number of visible points of kp2d in image when trunction.')
agt('--keep_truncation_dp', action='store_true',  help='keep points of densepose out of image when trunction.')
agt('--min_trunction_vis_dp_ratio',  default=0.0, type=float,  help='when dense points ratio in image less than this value, '
                                                                  'donot keep this label.')


agt('--coco_data_set', default='coco2014,coco2017',  help='0: donot use this data set.')
agt('--lsp_data_set',  default='lsp,lsp_ext', help='0: donot use this data set.')
agt('--mpii_data_set',  default='mpii', help='0: donot use this data set.')
agt('--hum36m_data_set', default='hum36m',  help='0: donot use this data set.')
agt('--pw3d_data_set', default='3dpw',  help='0: donot use this data set.')
agt('--coco_val_data_set', default='coco2017',  help='0: donot use this data set.')
agt('--hum36m_val_data_set', default='hum36m',  help='0: donot use this data set.')
agt('--pw3d_val_data_set', default='3dpw',  help='0: donot use this data set.')


## dataset coco
agt('--load_min_vis_kps', default=6, type=int, help='every coco picture shuold have one number of visible points at least.')


## dataset hum36m
agt('--hum36m_rot_prob', default=-1, type=float, help='.')
agt('--hum36m_rot_degree', default=25, type=float, help='.')
agt('--pw3d_rot_prob', default=-1, type=float, help='.')
agt('--pw3d_rot_degree', default=25, type=float, help='.')


opt = parser.parse_args()


################ network ################
opt.heads = {'box_hm':1, 'box_wh':2, 'box_cd':2, 'pose':72, 'shape':10, 'camera':3}
opt.smpl_cocoplus_path = os.path.join(root_dir, 'data', 'neutral_smpl_with_cocoplus_reg.pkl')
opt.smpl_basic_path = os.path.join(root_dir, 'data', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
opt.output_res = 128
opt.input_res = 512
opt.down_ratio = 4


################ dataset ################
# opt.coco_data_set=['coco2014', 'coco2017']
# opt.lsp_data_set=['lsp', 'lsp_ext']
# opt.hum36m_data_set=['hum36m']

# opt.coco_val_data_set=['coco2017']
# opt.hum36m_val_data_set=['hum36m']

# opt.train_adv_set = ['mosh']
# opt.eval_set = ['up3d']


opt.data_set_path = {
    'coco2014': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2014',
    'coco2017': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017',
    'lsp': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp/',
    # 'lsp_hr': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp_hr/',
    'lsp_ext': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp_extend',
    # 'lsp_ext_hr': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/lsp_extend_hr',
    # 'ai-ch':'E:/HMR/data/ai_challenger_keypoint_train_20170902',
    # 'mpi-inf-3dhp':'E:/HMR/data/mpi_inf_3dhp',
    'mpii': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/mpii',
    'hum36m': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/hum36m-toy',
    '3dpw': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/3DPW/',
    'mosh': 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/mosh/neutrMosh'
}

pre_process(opt)

################### preprocess ##################
# """log"""
# root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
# exp_dir = os.path.join(root_dir, 'exp')
# dir_name = opt.exp_id.replace(" ", "_")
# opt.save_dir = os.path.join(exp_dir, dir_name)
# opt.debug_dir = os.path.join(opt.save_dir, 'debug')
#
# """train"""
# opt.lr_step = [int(i) for i in opt.lr_step.split(',')]