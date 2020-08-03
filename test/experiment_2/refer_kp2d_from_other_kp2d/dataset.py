
import scipy.io as scio
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../../src')

import copy
import h5py
import json
import cv2
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import math

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.util import Clock, conver_crowdpose_to_cocoplus
from utils.opts import opt
from utils.debugger import Debugger

from utils.image import flip, color_aug
from utils.image  import get_affine_transform, affine_transform_bbox, affine_transform_kps, get_similarity_transform
from utils.image   import gaussian_radius, draw_umich_gaussian

np.random.seed(opt.data_aug_seed)


class crowdpose(Dataset):
    def __init__(self,
                data_path,
                image_path,
                split,
                max_data_len=-1):

        self.data_path = data_path
        self.image_path = image_path
        self.split = split
        self.max_data_len=max_data_len

        # load data set
        self._load_data_set()


    def _load_data_set(self):
        clk = Clock()
        print('==> loading crowdpose {} data.'.format(self.split))
        self.images = []

        self.img_dir = self.image_path
        anno_file_path = os.path.join(self.data_path, '{}.h5'.format(self.split))

        # key points
        with h5py.File(anno_file_path, 'r') as fp:
            self.kp2d_gt = np.array(fp['kp2d_gt'])
            self.kp2d_input = np.array(fp['kp2d_input'])

            for img_name in np.array(fp['imagename']):
                self.images.append(img_name.decode())

                if self.max_data_len > 0 and \
                        self.max_data_len <= len(self.images):
                    break

        print('loaded {} samples (t={:.2f}s)'.format(len(self.images), clk.elapsed()))


    def __len__(self):
        return len(self.images)


    def _get_image(self, index):
        img_name = self.images[index]
        img = cv2.imread(self.img_dir + '/' + img_name)

        return img


    def __getitem__(self, index):
        ## 1.get img
        # img = self._get_image(index)
        imagename = self.images[index]

        ## 2.input
        kp2d_input = self.kp2d_input[index][:,:,:2].flatten().astype(np.float32)
        kp2d = self.kp2d_gt[index].astype(np.float32)

        ## 3.
        # kp2d_gt = copy.deepcopy(kp2d)
        # kp2d_gt[:,0] = kp2d_gt[:,0] * img.shape[1]
        # kp2d_gt[:,1] = kp2d_gt[:,1] * img.shape[0]

        gt = []
        gt.append({
            'kp2d': kp2d,
            'kp2d_input':self.kp2d_input[index]
        })

        return {
            'imagepath': self.img_dir + '/' + imagename,
            'kp2d_input': kp2d_input,
            'kp2d': kp2d,
            'gt': gt
        }


if __name__ == '__main__':
    torch.manual_seed(opt.seed)

    # image_path = 'F:\paper\dataset\crowdpose'
    image_path = '/home/icvhpc1/bluce/dataset/crowdpose'

    data = crowdpose(
        data_path = abspath + '//data',
        image_path = image_path,
        split = 'train',
        max_data_len=10
    )

    data_loader = DataLoader(data, batch_size=2, shuffle=False)
    debugger = Debugger(opt.smpl_basic_path,
                        opt.smpl_cocoplus_path,
                        'cocoplus')

    for batch in data_loader:
        imagepath = batch['imagepath'][0]
        img = cv2.imread(imagepath)

        h, w = img.shape[0:2]

        # 1.
        input_id = 'kp2d_input'
        debugger.add_img(img, img_id=input_id)

        for kp2d in batch['gt'][0]['kp2d_input'][0]:
            kp2d = kp2d.detach().cpu().numpy()
            kp2d[:, 0] = kp2d[:, 0] * w
            kp2d[:, 1] = kp2d[:, 1] * h

            kp2d = conver_crowdpose_to_cocoplus(kp2d)
            debugger.add_kp2d(kp2d, img_id=input_id)

        # 2.
        kp2d_id = 'kp2d'
        debugger.add_img(img, img_id=kp2d_id)

        kp2d = batch['gt'][0]['kp2d'][0].detach().cpu().numpy()
        kp2d[:, 0] = kp2d[:, 0] * w
        kp2d[:, 1] = kp2d[:, 1] * h

        kp2d = conver_crowdpose_to_cocoplus(kp2d)
        debugger.add_kp2d(kp2d, img_id=kp2d_id)


        debugger.show_all_imgs(pause=True)