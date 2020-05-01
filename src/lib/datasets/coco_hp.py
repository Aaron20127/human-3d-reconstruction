from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../trains')
sys.path.insert(0, abspath + '/../utils')

import argparse

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np

import torch.utils.data as data

import json
import cv2

from image import flip, color_aug
from image import get_affine_transform, affine_transform, affine_transform_bbox
from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from image import draw_dense_reg
import math

from util import addCocoAnns

class COCOHP(data.Dataset):
    num_classes = 1
    num_joints = 17
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]

    def __init__(self, opt, split):
        super(COCOHP, self).__init__()
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [4, 6], [3, 5], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [6, 12], [5, 11], [11, 12],
                      [12, 14], [14, 16], [11, 13], [13, 15]]

        self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.data_dir = opt.data_dir
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'person_keypoints_{}2017.json').format(split)
        self.max_objs = 32 # max number of objects in one image
        self._data_rng = np.random.RandomState(123) # for data color augment
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32) # for data color augment
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32) # for data color augment
        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        # person and not crowd
        self.images = []
        for img_id in image_ids: # only save the image ids who have annotations
            idxs = self.coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
            if len(idxs) > 0:
                self.images.append(img_id)

        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))


    def __len__(self):
        return self.num_samples


    def run_eval(self, results, save_dir):
        def _to_float(x):
            return float("{:.2f}".format(x))

        def convert_eval_format(all_bboxes):
            # import pdb; pdb.set_trace()
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out = list(map(_to_float, bbox))
                        keypoints = np.concatenate([
                            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
                            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                        keypoints = list(map(_to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "keypoints": keypoints
                        }
                        detections.append(detection)
            return detections

        def save_results(results, save_dir):
            json.dump(convert_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


    def _get_image(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0) # remove crowd annotations
        anns = self.coco.loadAnns(ids=ann_ids)
        img = cv2.imread(img_path)

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

        opt = self.opt
        split = self.split
        _data_rng = self._data_rng
        _eig_val = self._eig_val
        _eig_vec = self._eig_vec
        mean = self.mean
        std = self.std

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # image center
        s = max(img.shape[1], img.shape[0]) * 1.0  # max img length
        rot = 0
        flipped = False
        if split == 'train':
            if not opt.not_rand_crop:  # random crop, get center and scale
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                # to make sure that we can get a random cneter point, namely img.shape > 2*border
                w_border = _get_border(128, img.shape[1])
                h_border = _get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = opt.scale
                cf = opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < opt.aug_rot:  # whether or not to rotate
                rf = opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < opt.flip:  # flip
                flipped = True
                img = img[:, ::-1, :]
                c[0] = img.shape[1] - c[0] - 1

            # use affine transform to scale, rotate and crop image
        trans_input = get_affine_transform(
            c, s, rot, [opt.input_res, opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)

        # normalize, color augment and standardize image
        inp = (inp.astype(np.float32) / 255.)  # normalize
        if not opt.no_color_aug:  # color augment
            color_aug(_data_rng, inp, _eig_val, _eig_vec)
        inp = (inp - mean) / std  # standardize
        inp = inp.transpose(2, 0, 1)  # change channel (3, 512, 512)

        return inp, c, s, rot, flipped


    def _get_label(self, c, s, rot, width, flipped, anns):
        def _coco_box_to_bbox(box):
            bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                            dtype=np.float32)
            return bbox

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res]) # affine transform  to 128x128 with rotation
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res]) # affine transform to 128x128 without rotation

        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                             dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                                  dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        num_objs = min(len(anns), self.max_objs) # max number of objects, default 32
        for k in range(num_objs):
            ann = anns[k]
            bbox = _coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            ## affine transform bbox to feature map 128x128
            # bbox[:2] = affine_transform(bbox[:2], trans_output)
            # bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = affine_transform_bbox(bbox, trans_output_rot)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                ## 3.1 handle pure bbox
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) # center of bbox
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h  # width and height of bbox
                ind[k] = ct_int[1] * output_res + ct_int[0] # center of bbox in feature map index 0-16384
                reg[k] = ct - ct_int # decimal of center of bbox
                reg_mask[k] = 1 # center mask ???
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0: # if no key points, hm=0.9999, reg_mask[k]=0
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0

                ## 3.2
                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    if pts[j, 2] > 0: # key points is visible
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                pts[j, 1] >= 0 and pts[j, 1] < output_res: # key points in output feature map
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int # vector of key points to cneter of bbox
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            if self.opt.dense_hp:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                               pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([bbox[0], bbox[1], bbox[2], bbox[3], 1] + # [0:4] bbox，[4] 1, [5:39] key points, [40] class id 0
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0

        return gt_det, hm, reg, reg_mask, ind, wh, kps, kps_mask, hm_hp, \
               hp_offset, hp_ind, hp_mask, dense_kps, dense_kps_mask


    def _get_dataset(self,  inp, hm, reg, reg_mask, ind, wh,
                            kps, kps_mask, hm_hp, hp_offset, hp_ind, hp_mask,
                            c, s, gt_det, img_id):
        """
        :param inp: input of network. (3, 512, 512)
        :param hm:  heat map of bbox center. (1, 128, 128)
        :param reg: decaimal of bbox center. (number_object, 2)
        :param reg_mask: decaimal mask of bbox center. (number_object, )
        :param ind: index of bbox center in feature map. 0-128*128. (number_object, )
        :param wh: hieght and width of bbox. (number_object, 2)
        :param kps: vector of key points to bbox center. (number_object, 2*17)
        :param kps_mask: vector mask of key points to bbox center. (number_object, 2*17)
        :param hm_hp: heat map of key points. (17, 128, 128)
        :param hp_offset: decimal of key points center. (number_object * 17, 2)
        :param hp_ind: index of key points. (number_object * 17, )
        :param hp_mask: key points mask. (number_object * 17, )
        :param c: random position of input image as center to image cut.
        :param s: scale for image cut.
        :param gt_det: ground truth (40, ). [0:4] bbox, [4] 1, [5:39] key points, [39] class id 0.
        :param img_id: image id in coco dataset.

        :return: training data.
        """
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask}
        num_joints = self.num_joints
        output_res = self.opt.output_res

        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret


    def __getitem__(self, index):
        ## 1.get img and anns
        img, anns, img_id = self._get_image(index)

        # img_anns = addCocoAnns(anns, img)
        # cv2.imshow(str(img_id),img_anns)
        # cv2.waitKey(0)
        # cv2.destroyWindow(str(img_id))

        ## 2. handle input of network to 512x512, namely crop and normalize image
        inp, c, s, rot, flipped = self._get_input(img)

        ## 3. handle output of network, namely label
        width = img.shape[1]
        gt_det, hm, reg, reg_mask, ind, wh, kps, kps_mask, hm_hp, \
        hp_offset, hp_ind, hp_mask, dense_kps, dense_kps_mask =\
            self._get_label(c, s, rot, width, flipped, anns)

        ## 4. assemble label
        ret = self._get_dataset(
                    inp, hm, reg, reg_mask, ind, wh, \
                    kps, kps_mask, hm_hp, hp_offset, hp_ind, hp_mask,\
                    c, s, gt_det, img_id)
        return ret



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