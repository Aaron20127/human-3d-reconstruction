
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import json
import cv2
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import math

import torch.utils.data as data

from utils.util import Clock
from utils import opts

from utils.image import flip, color_aug

from utils.image  import get_affine_transform, affine_transform_bbox, affine_transform_kps
from utils.image   import gaussian_radius, draw_umich_gaussian
from utils.image   import draw_dense_reg
from utils.image  import addCocoAnns


class COCO2017(data.Dataset):
    def __init__(self,
                data_path,
                scale_range=(0.6, 1.4),
                flip_prob=0.5,
                rot_prob=0.0,
                rot_degree=np.pi/4,
                output_res=512,
                max_objs = 32,
                split = 'train'):

        self.data_path = data_path
        self.max_objs = max_objs  # max number of objects in one image
        self.split = split
        self.output_res = output_res
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.rot_degree = rot_degree

        # defaut parameters
        self.num_joints = 19
        self.kps_map = [16, 14, 12, 11, 13, 15, 10, 8, 6,
                        5, 7, 9, 0, 0, 0, 1, 2, 3, 4]  # key points map coco to smpl cocoplus key points
        self.flip_idx = [[0, 5], [1, 4], [2, 3], [8, 9], [7, 10],
                         [6, 11], [15, 16], [17, 18]] # smpl cocoplus key points flip index

        # load data set
        self._load_data_set()


    def _load_data_set(self):
        clk = Clock()
        print('=> start load coco2017 {} data.'.format(self.split))
        self.image_ids = []

        self.img_dir = os.path.join(self.data_path, '{}2017'.format(self.split))
        if self.split == 'eval':
            self.annot_path = os.path.join(
                self.data_path, 'annotations',
                'image_info_test-dev2017.json').format(self.split)
        else:
            self.annot_path = os.path.join(
                self.data_path, 'annotations',
                'person_keypoints_{}2017.json').format(self.split)
        self.coco = coco.COCO(self.annot_path)

        # person and not crowd
        ids = self.coco.getImgIds()
        for img_id in ids:  # only save the image ids who have annotations
            idxs = self.coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
            if len(idxs) > 0:
                self.img_ids.append(img_id)

        print('loaded {} samples, time elapsed {}.'.format(len(self.img_ids), clk.elapesd()))


    def __len__(self):
        return len(self.img_ids)


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


    def _get_input(self, img):
        def _get_border(border, size):
            """
            :param border: 0 <= border <= 128, size > 2*border, then we can get a rander center in central area
            :param size: image height and width
            :return: min border
            """
            i = 1
            while size - border // i <= border // i:
                i *= 2
            return border // i


        c = np.array([img.shape[1] / 2., img.shape[0] / 2.])  # image center
        s = max(img.shape[1], img.shape[0]) * 1.0  # max img length

        rot = 0
        flipped = False
        if self.split == 'train':
            if not opt.not_rand_crop:  # random crop, get center and scale
                s = s * np.random.choice(np.arange(self.scale_range[0], self.scale_range[1], 0.1))
                # to make sure that we can get a random cneter point, namely img.shape > 2*border
                w_border = _get_border(128, img.shape[1])
                h_border = _get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if np.random.random() < self.rot_prob:  # whether or not to rotate
                rf = self.rot_degree
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < opt.flip:  # flip
                flipped = True
                img = img[:, ::-1, :]
                c[0] = img.shape[1] - c[0] - 1

        # use affine transform to scale, rotate and crop image
        trans_input = get_affine_transform(
                      c, s, rot, [self.output_res, self.output_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.output_res, self.output_res),
                             flags=cv2.INTER_LINEAR)

        # normalize, color augment and standardize image
        # inp = (inp.astype(np.float32) / 255.)  # normalize
        # if not self.no_color_aug:  # color augment
        #     color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # inp = inp / 255. * 2.0 - 1.0  # standardize

        inp = (inp.astype(np.float) / 255) * 2.0 - 1.0  # normalize
        inp = inp.transpose(2, 0, 1)  # change channel (3, 512, 512)

        return inp, c, s, rot, flipped


    def _convert_kps_coco_to_smpl(self, pts):
        """
         covert coco key pints to smpl cocoplus key points.

         Argument
            coco_pts (array, (17,3)): coco key points list.
        """
        kps = pts[self.kps_map].copy()
        kps[12:14] = 0  # no neck, top head
        kps[:, 2] = kps[:, 2] > 0  # visible points to be 1 # TODO debug
        return kps


    def _get_bbox(self, bbox, flipped, width, affine_mat):
        """
         create objcet label.

         Argument
            bbox (list): coco bbox .
            flipped (bool): whether has flipped.
            width (int): Original image width.
            affine_mat (array, (2,3)): matrix of affine trans.
        """
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2],
                         bbox[1] + bbox[3]], dtype=np.float32)

        if flipped:
            bbox[[0, 2]] = width - bbox[[2, 0]] - 1

        ## bbox transform
        bbox = affine_transform_bbox(bbox, affine_mat)
        bbox = np.clip(bbox, 0, self.output_res - 1)  # save the bbox in the image

        return bbox


    def _get_kps(self, kps, flipped, width, affine_mat):
        kps = np.array(kps).reshape(-1, 3)

        # convert key points serial number
        kps = self._convert_kps_coco_to_smpl(kps)

        # flip
        if flipped:
            kps[:, 0] = width - kps[:, 0] - 1  # points mirror
            for e in self.flip_idx:
                kps[e[0]], kps[e[1]] = kps[e[1]].copy(), kps[e[0]].copy()  # key points name mirror

        # affine transform
        kps = affine_transform_kps(kps, affine_mat)

        return kps


    def _get_label(self, c, s, rot, width, flipped, anns):
        box_hm = np.zeros((self.output_res, self.output_res),dtype=np.float32)

        box_ind = np.zeros((self.max_objs), dtype=np.int64)
        box_mask = np.zeros((self.max_objs), dtype=np.uint8)
        box_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        box_cd = np.zeros((self.max_objs, 2), dtype=np.float32) # bbox center decimal

        kp2d_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kp2d = np.zeros((self.max_objs, self.num_joints * 3), dtype=np.float32)

        gt = []

        # draw heap map function
        draw_gaussian = draw_umich_gaussian

        # affine transform  to crop, scale, rotate
        affine_mat = get_affine_transform(c, s, rot, \
                                [self.output_res, self.output_res])

        num_objs = min(len(anns), self.max_objs) # max number of objects
        for k in range(num_objs):
            ann = anns[k]

            ### 1. bbox
            bbox = self._get_bbox(ann['bbox'], flipped, width, affine_mat)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h > 0 and w > 0):  # if outside the image, discard
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])  # center of bbox
                ct_int = ct.astype(np.int32)

                box_wh[k] = 1. * w, 1. * h  # width and height of bbox
                box_ind[k] = ct_int[1] * self.output_res + ct_int[0]  # center of bbox in feature map index 0-16384
                box_cd[k] = ct - ct_int  # decimal of center of bbox
                box_mask[k] = 1  # box ind mask

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                draw_gaussian(box_hm, ct_int, radius)  # draw heat map


            ### 2.handle key points
            kps = self._get_kps(ann['keypoints'], flipped, width, affine_mat)

            vis_kps = 0
            for j in range(self.num_joints):
                if kps[j, 2] > 0: # key points is visible
                    if kps[j, 0] >= 0 and kps[j, 0] < self.output_res and \
                       kps[j, 1] >= 0 and kps[j, 1] < self.output_res: # key points in output feature map
                       real_vis_kps += 1
                       kp2d[k, j] = kps[j]
            if vis_kps > self.min_vis_kps:
                kp2d_mask[k] = 1

            ### 3. groud truth
            gt.append([bbox, kps])

            return box_hm, box_wh, box_cd, box_ind, box_mask, kp2d, kp2d_mask, gt


    def __getitem__(self, index):
        """
        return: {
                    'hm':           '(n, 1, 128, 128)',  # bbox center heat map

                    'wh_ind':       '(n, max_obj)', # bbox width, height
                    'wh_mask':      '(n, max_obj)',
                    'wh':           '(n, max_obj, 2)',

                    'theta_ind':    '(n, max_obj)'
                    'theta_mask':   '(n, max_obj)'
                    'pose':         '(n, max_obj, 72)',
                    'shape':        '(n, max_obj, 10)',

                    'kp_2d_ind':    '(n, max_obj)'
                    'kp_3d_mask':   '(n, max_obj)'
                    'kp_3d':        '(n, 19, 3)',

                    'kp_2d_ind':    '(n, max_obj)'
                    'kp_3d_mask':   '(n, max_obj)'
                    'kp_2d':        '(n, 19, 3)', # 第三列是否可见可以作为索引，加上coco数据集的眼睛、耳朵和鼻子

                    'dataset':      'coco2017'
                }
        """
        ## 1.get img and anns
        img, anns, img_id = self._get_image(index)

        ## 2. handle input of network to 512x512, namely crop and normalize image
        inp, c, s, rot, flipped = self._get_input(img)

        ## 3. handle output of network, namely label
        width = img.shape[1]
        box_hm, box_wh, box_cd, box_ind, box_mask, kp2d, kp2d_mask, gt =\
            self._get_label(c, s, rot, width, flipped, anns)

        return {
            'input': inp,
            'box_hm': box_hm,
            'box_wh': box_wh,
            'box_cd': box_cd,
            'box_ind': box_ind,
            'box_mask': box_mask,
            'kp2d': kp2d,
            'kp2d_mask': kp2d_mask,
            'gt': gt,
            'dataset': 'coco2017'
        }

if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.data_dir = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017'
    opt.not_rand_crop = False
    opt.aug_rot = 0
    opt.scale = 0.4
    opt.shift = 0.1
    opt.rotate = 0
    opt.flip = 0.5
    opt.input_res = 512
    opt.output_res = 512
    opt.no_color_aug = False
    opt.mse_loss = False
    opt.dense_hp = False
    opt.reg_offset = True
    opt.hm_hp = True
    opt.reg_hp_offset = True
    opt.debug = 0
    opt.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    opt.std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)

    val_loader = data.DataLoader(
        COCOHP(opt, 'val'), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = data.DataLoader(
        COCOHP(opt, 'train'), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    data_loder = train_loader

    for i, batch in enumerate(data_loder):
        img = batch['input'][0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
        cv2.imshow('img', img)
        cv2.waitKey(0)