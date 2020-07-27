import h5py
import copy
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../../src')

import json
import cv2
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.util import Clock, decode_label_bbox, decode_label_kp2d
from utils.opts import opt
from utils.debugger import Debugger

from utils.image import flip, color_aug
from utils.image  import get_affine_transform, affine_transform_bbox, affine_transform_kps, get_similarity_transform
from utils.image   import gaussian_radius, draw_umich_gaussian
from utils.image   import draw_dense_reg
from utils.image  import addCocoAnns

np.random.seed(opt.data_aug_seed)

class crowdpose():
    def __init__(self,
                 data_path,
                 save_path,
                 min_kps=10,
                 min_num_person=2,
                 max_num_person=10,
                 num_kps = 17,
                 split='train',
                 max_data_len=-1):

        self.data_path = data_path
        self.save_path = save_path
        self.split = split
        self.min_kps=min_kps
        self.min_num_person = min_num_person
        self.max_data_len = max_data_len
        self.max_num_person = max_num_person
        self.num_kps = num_kps

        # load data set
        self._load_data_set()


    def _load_data_set(self):
        clk = Clock()
        print('==> loading crowdpose {} data.'.format(self.split))

        self.img_dir = os.path.join(self.data_path, 'images'.format(self.split))
        if self.split == 'test':
            self.annot_path = os.path.join(
                self.data_path, 'annotations',
                'crowdpose_test.json')
        else: # train or val
            self.annot_path = os.path.join(
                self.data_path, 'annotations',
                'crowdpose_{}.json').format(self.split)
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        # person and not crowd
        self.images = []
        for img_id in image_ids: # only save the image ids who have annotations
            idxs = self.coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
            anns = self.coco.loadAnns(ids=idxs)

            num_person = 0
            for ann in anns:
                kp = np.array(ann['keypoints']).reshape(-1, 3)
                num_valid_kp = np.sum(kp[:, 2] > 0)
                if num_valid_kp >= self.min_kps:
                    num_person += 1
                    if num_person >= self.min_num_person:
                        self.images.append(img_id)
                        break

            if  self.max_data_len > 0 and \
                self.max_data_len <= len(self.images):
                break

        print('loaded {} samples (t={:.2f}s)'.format(len(self.images), clk.elapsed()))


    def _get_image(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0) # remove crowd annotations
        anns = self.coco.loadAnns(ids=ann_ids)

        img = cv2.imread(img_path)
        # import jpeg4py as jpeg
        # img = jpeg.JPEG(img_path).decode() # accelerate jpeg image read speed

        return img, anns, img_id


    def _pack_rest_data(self, k, input_list):
        ret = np.zeros((self.max_num_person-1, self.num_kps, 3))
        inp = input_list[k]

        sort_list = []
        for i, kp in enumerate(input_list):
            if i != k:
                sort_list.append((inp, kp))

        def distanse(elem):
            mask = ((elem[0][:,2] > 0).astype(np.int) + (elem[1][:,2] > 0).astype(np.int)) > 1
            return np.sum((elem[0][mask, 0:2] - elem[1][mask, 0:2])**(2)) / np.sum(mask)

        sort_list_new = sorted(sort_list, key = distanse)

        for j, elem in enumerate(sort_list_new):
            if j >= self.max_num_person-1:
                break
            kp = elem[1]
            ret[j, :, :2] = kp[:, :2]
            ret[j, :, 2] = (kp[:, 2] > 0).astype(np.int)

        return ret



    def save_data(self):
        _kp2d_input = np.zeros((1000000, self.max_num_person, self.num_kps, 3))
        _kp2d_gt = np.zeros((1000000, self.num_kps, 3))
        _imagename = []
        num = 0

        for index in range(len(self.images)):

            ## 1.get img and anns
            img, anns_coco, img_id = self._get_image(index)


            ## 2. handle output of network, namely label
            input_list = []

            for ann in anns_coco:
                kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float)
                kp[:, 0] = kp[:, 0] / img.shape[1]  # 归一化坐标
                kp[:, 1] = kp[:, 1] / img.shape[0]

                input_list.append(kp)

            #
            print('{} / {}, {}'.format(index, len(self.images), len(input_list)))

            ## 3.

            for k, kp in enumerate(input_list):
                if np.sum(kp[:, 2] > 0) > self.min_kps:
                    for i in range(kp.shape[0]):
                        if kp[i,2] > 0:
                            _kp2d_input[num, 0, :, :2] = kp[:, :2]
                            _kp2d_input[num, 0, :, 2] = (kp[:, 2] > 0).astype(np.int)
                            _kp2d_input[num, 0, i, :] = 0

                            _kp2d_input[num, 1:self.max_num_person, :, :] = self._pack_rest_data(k, input_list)

                            _kp2d_gt[num, :, :2] = kp[:, :2]
                            _kp2d_gt[num, :, 2] = (kp[:, 2] > 0).astype(np.int)
                            _imagename.append('images/' + str(img_id) + '.jpg')

                            num += 1


        dst_file = os.path.join(self.save_path, self.split+'.h5')
        dst_fp = h5py.File(dst_file, 'w')

        dst_fp.create_dataset('kp2d_input', data=_kp2d_input[:num])
        dst_fp.create_dataset('kp2d_gt', data=_kp2d_gt[:num])
        dst_fp.create_dataset('imagename', data=np.array(_imagename[:num], dtype='S'))
        dst_fp.close()

        print('done, total {}, '.format(num))


if __name__ == '__main__':
    data = crowdpose( 'F:\\paper\\dataset\\crowdpose',
                    'G:\\paper\\code\\master\\test\\experiment_2\\refer_kp2d_from_other_kp2d\\data',
                    min_kps = 10,
                    min_num_person = 2,
                    max_num_person = 10,
                    num_kps = 14,
                    split = 'val',
                    max_data_len = -1)

    data.save_data()