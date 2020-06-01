
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


import cv2
import numpy as np
import math
import h5py


from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.util import Clock, reflect_pose, decode_label_bbox, decode_label_kp2d, get_camera_from_batch
from utils.opts import opt
from utils.debugger import Debugger

from utils.image import flip, color_aug
from utils.image  import get_affine_transform, affine_transform_bbox, affine_transform_kps, get_similarity_transform
from utils.image   import gaussian_radius, draw_umich_gaussian
from utils.image   import draw_dense_reg
from utils.image  import addCocoAnns

np.random.seed(opt.data_aug_seed)

class PW3D(Dataset):
    def __init__(self,
                data_path,
                image_scale_range=(0.6, 1.4),
                trans_scale=1,
                flip_prob=0.5,
                rot_prob=0.0,
                rot_degree=0.0,
                color_aug=True,
                input_res=512,
                output_res=128,
                max_objs = 32,
                split = 'train', # train, val, test
                min_vis_kps = 6,
                keep_truncation_kps=False,
                min_truncation_kps=12,
                min_truncation_kps_in_image=6,
                normalize = True,
                box_stretch = 20,
                min_bbox_area = 16*16,
                max_data_len = -1):

        self.data_path = data_path
        self.image_scale_range = image_scale_range
        self.trans_scale = trans_scale
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.rot_degree = rot_degree
        self.color_aug = color_aug
        self.input_res = input_res
        self.output_res = output_res
        self.max_objs = max_objs
        self.split = split  # train, val, test
        self.min_vis_kps = min_vis_kps
        self.normalize = normalize
        self.box_stretch = box_stretch
        self.down_ratio = input_res / output_res
        self.max_data_len = max_data_len
        self.keep_truncation_kps = keep_truncation_kps
        self.min_truncation_kps_in_image = min_truncation_kps_in_image
        self.min_truncation_kps = min_truncation_kps
        self.min_bbox_area = min_bbox_area

        # defaut parameters
        # key points
        self.num_joints = 19
        self.kps_map = [10,9,8,11,12,13,4,3,2,5,6,7,1,0,0,15,14,17,16]  # key points map lsp to smpl cocoplus key points
        self.not_exist_kps = [13]
        self.flip_idx = [[0, 5], [1, 4], [2, 3], [8, 9], [7, 10],
                         [6, 11], [15, 16], [17, 18]] # smpl cocoplus key points flip index

        # load data set
        self._load_data_set()


    def _load_data_set(self):
        clk = Clock()
        print('==> loading 3dpw {} data.'.format(self.split))
        self.images = []

        if self.split == 'train':
            anno_file_path = os.path.join(self.data_path, 'annotations', 'train.h5')
        if self.split == 'val':
            anno_file_path = os.path.join(self.data_path, 'annotations', 'validation.h5')

        with h5py.File(anno_file_path, 'r') as fp:
            self.kp2ds = np.array(fp['gt2d'])
            self.shape = np.array(fp['shape'])
            self.pose = np.array(fp['pose'])

            self.kp2ds[..., 2] = (self.kp2ds[..., 2] > 0).astype(np.float32)

            for img_name in np.array(fp['imagename']):
                self.images.append(img_name.decode())

                if self.max_data_len > 0 and \
                   self.max_data_len <= len(self.images):
                    break

            self.img_dir = os.path.join(self.data_path)

        print('loaded {} samples (t={:.2f}s)'.format(len(self.images), clk.elapsed()))


    def __len__(self):
        return len(self.images)


    def _get_image(self, index):
        img_name = self.images[index]
        img = cv2.imread(self.img_dir + '/' + img_name)
        # import jpeg4py as jpeg
        # img = jpeg.JPEG(img_path).decode() # accelerate jpeg image read speed

        return img


    def _get_input(self, img):
        h, w = img.shape[0], img.shape[1]
        s = self.input_res * 1.0 / max(w, h)  # defalut scale
        t = np.array([self.input_res / 2., self.input_res / 2.])  # translate to image center
        r = 0
        rand_scale = 1
        flip = False

        if self.split == 'train':
            ## scale
            rand_scale = np.random.choice(np.arange(
                self.image_scale_range[0], self.image_scale_range[1], 0.1))
            s = s * rand_scale

            ## translate
            t[0] = t[0] + self.trans_scale * (np.random.random() * 2 - 1) \
                   * (self.input_res / 2.0 + s * w / 2.0)
            t[1] = t[1] + self.trans_scale * (np.random.random() * 2 - 1) \
                   * (self.input_res / 2.0 + s * h / 2.0)

            if self.rot_prob > np.random.random():  # whether or not to rotate
                r = (np.random.random() * 2 - 1) * self.rot_degree

            if self.flip_prob > np.random.random():  # flip
                flip = True
                # img = img[:, ::-1, :]
                # c[0] = img.shape[1] - c[0] - 1

        # use affine transform to scale, rotate and crop image
        trans_mat = get_similarity_transform(s, t, r, flip, w, h)
        inp = cv2.warpAffine(img, trans_mat,
                             (self.input_res, self.input_res),
                             flags=cv2.INTER_LINEAR)

        # cv2.imshow('img', img)
        # cv2.imshow('inp', cv2.resize(inp.astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC))
        # cv2.waitKey(0)

        # normalize, color augment and standardize image
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and \
           self.color_aug:  # color augment
            color_aug(inp)

        if self.normalize:
            inp = inp * 2.0 - 1.0  # normalize
        else:
            inp = inp * 255

        inp = inp.transpose(2, 0, 1)  # change channel (3, 512, 512)

        return inp, trans_mat, flip, rand_scale


    def _convert_kp2d_to_smpl(self, pts):
        """
         covert coco key pints to smpl cocoplus key points.

         Argument
            pts (array, (14,3)): lsp 2d key points list.
        """
        kps = pts[self.kps_map].copy()
        kps[self.not_exist_kps] = 0
        kps[:, 2] = kps[:, 2] > 0  # visible points to be 1 # TODO debug
        return kps


    def _convert_kp3d_to_smpl(self, pts):
        """
         covert coco key pints to smpl cocoplus key points.

         Argument
             pts (array, (14,3)): lsp 3d key points list.
        """
        kps = pts[self.kps_map].copy()
        kps[self.not_exist_kps] = 0
        return kps


    def _generate_bbox(self, kp, flip, affine_mat, rand_scale):  # TODO use object detection to get bbox
        kp = self._get_kp_2d(kp, flip, affine_mat)
        box_stretch = rand_scale * self.box_stretch

        v_kp = kp[kp[:, 2] > 0]
        x_min = v_kp[:, 0].min()
        x_max = v_kp[:, 0].max()
        y_min = v_kp[:, 1].min()
        y_max = v_kp[:, 1].max()

        x_l = x_min - box_stretch if x_min - box_stretch > 0 else 0
        y_l = y_min - box_stretch if y_min - box_stretch > 0 else 0  # head special handle
        x_r = x_max + box_stretch if x_max + box_stretch < self.input_res - 1 else self.input_res - 1
        y_r = y_max + box_stretch if y_max + box_stretch < self.input_res - 1 else self.input_res - 1

        coco_bbox = [x_l,
                     y_l,
                     x_r - x_l,
                     y_r - y_l]

        return coco_bbox

    def _get_bbox(self, bbox, affine_mat):
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

        ## bbox transform
        # bbox = affine_transform_bbox(bbox, affine_mat)  # auto flip
        # bbox = np.clip(bbox, 0, self.input_res - 1)  # save the bbox in the image

        return bbox

    def _get_kp_2d(self, kps, flipped, affine_mat):
        # convert key points serial number
        kps = self._convert_kp2d_to_smpl(kps)

        # flip
        if flipped:
            for e in self.flip_idx:
                kps[e[0]], kps[e[1]] = kps[e[1]].copy(), kps[e[0]].copy()  # key points name mirror

        # affine transform
        kps = affine_transform_kps(kps, affine_mat)

        return kps

    def _get_kp_3d(self, kps, flipped):
        # convert key points serial number
        # kps = self._convert_kp3d_to_smpl(kps)

        # flip
        if flipped: # TODO whether need to
            for e in self.flip_idx:
                kps[e[0]], kps[e[1]] = kps[e[1]].copy(), kps[e[0]].copy()  # key points name mirror

            kps[:, 0] = -kps[:, 0]

        return kps


    def _get_pose(self, pose, flipped):

        # flip
        if flipped: # TODO whether need to
            pose = reflect_pose(pose.flatten())
        return pose


    def _get_label(self, trans_mat, flip, anns):
        box_hm = np.zeros((1,self.output_res, self.output_res), dtype=np.float32)

        box_ind = np.zeros((self.max_objs), dtype=np.int64)
        box_mask = np.zeros((self.max_objs), dtype=np.uint8)
        box_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        box_cd = np.zeros((self.max_objs, 2), dtype=np.float32)  # bbox center decimal

        kp2d_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kp2d = np.zeros((self.max_objs, self.num_joints, 3), dtype=np.float32)

        kp3d_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kp3d = np.zeros((self.max_objs, self.num_joints, 3), dtype=np.float32)
        has_kp3d = np.array([0], dtype=np.uint8)

        has_theta = np.array([1], dtype=np.uint8)
        theta_mask = np.zeros((self.max_objs), dtype=np.uint8)
        shape = np.zeros((self.max_objs, 10), dtype=np.float32)
        pose = np.zeros((self.max_objs, 72), dtype=np.float32)

        has_dp = np.array([0], dtype=np.uint8)

        gt = []

        # draw heap map function
        draw_gaussian = draw_umich_gaussian
        num_objs = min(len(anns), self.max_objs)  # max number of objects
        for k in range(num_objs):
            ann = anns[k]

            bbox = self._get_bbox(ann['bbox'], trans_mat) / self.down_ratio
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h > 0 and w > 0):  # if outside the image, discard
                if h*w*self.down_ratio <= self.min_bbox_area:
                    # print( h*w*self.down_ratio, self.min_bbox_area)
                    continue

                ### 1. handle bbox
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]) # down ratio
                ct_int = ct.astype(np.int32)

                box_wh[k] = 1. * w, 1. * h# width and height of bbox
                box_ind[k] = ct_int[1] * self.output_res + ct_int[0]  # center of bbox in feature map index 0-16384
                box_cd[k] = ct - ct_int  # decimal of center of bbox
                box_mask[k] = 1  # box ind mask

                radius = gaussian_radius((math.ceil(h),
                                          math.ceil(w)))
                radius = max(0, int(radius))
                draw_gaussian(box_hm[0], ct_int, radius)  # draw heat map


                ### 2.handle 2d key points
                kps = self._get_kp_2d(ann['kp2d'], flip, trans_mat)

                total_kps = kps[:, 2].sum()
                if total_kps >= self.min_vis_kps:
                    vis_kps = 0
                    for j in range(self.num_joints):
                        if kps[j, 2] > 0:  # key points is visible
                            if kps[j, 0] >= 0 and kps[j, 0] < self.input_res and \
                                    kps[j, 1] >= 0 and kps[j, 1] < self.input_res:  # key points in output feature map
                                vis_kps += 1
                                kp2d[k, j] = kps[j]

                    if vis_kps >= self.min_vis_kps:
                        if total_kps != vis_kps:
                            if self.keep_truncation_kps == True and \
                                    self.min_truncation_kps <= total_kps and \
                                    self.min_truncation_kps_in_image <= vis_kps:
                                kp2d[k] = kps

                        kp2d_mask[k] = 1


                ### 3. handle 3d key points
                # kp3d[k] = self._get_kp_3d(ann['kp3d'], flip)
                # kp3d_mask[k] = 1


                ### 4. handle pose and shape
                pose[k] = self._get_pose(ann['pose'], flip)
                shape[k] = ann['shape']
                theta_mask[k] = 1


                ### groud truth
                gt.append({
                    'bbox': bbox * self.down_ratio,
                    'kp2d': kp2d[k],
                    'kp3d': kp3d[k],
                    'pose': pose[k],
                    'shape': shape[k]
                })

        return box_hm, box_wh, box_cd, box_ind, box_mask, kp2d, kp2d_mask, \
               theta_mask, pose, shape, has_theta, kp3d, kp3d_mask, has_kp3d, has_dp, gt



    def __getitem__(self, index):

        ## 1.get img and anns
        img = self._get_image(index)

        ## 2. handle input of network to 512x512, namely crop and normalize image
        inp, trans_mat, flip, rand_scale = self._get_input(img)

        ## 3. handle output of network, namely label
        anns = []
        for i, kp2d in enumerate(self.kp2ds[index]):
            if kp2d[:,2].sum() <= 0:
                continue

            coco_bbox = self._generate_bbox(kp2d, flip, trans_mat, rand_scale)
            anns.append({
                'bbox': coco_bbox,
                'kp2d': kp2d,
                'shape': self.shape[index][i],
                'pose':  self.pose[index][i]
            })

        box_hm, box_wh, box_cd, box_ind, box_mask, kp2d, kp2d_mask, \
        theta_mask, pose, shape, has_theta, kp3d, kp3d_mask, has_kp3d, has_dp, gt = \
            self._get_label(trans_mat, flip, anns)


        return {
            'input': inp,
            'box_hm': box_hm,
            'box_wh': box_wh,
            'box_cd': box_cd,
            'box_ind': box_ind,
            'box_mask': box_mask,
            'kp2d': kp2d,
            'kp2d_mask': kp2d_mask,
            'kp3d': kp3d,
            'kp3d_mask': kp3d_mask,
            'has_kp3d': has_kp3d,
            'pose': pose,
            'shape': shape,
            'theta_mask': theta_mask,
            'has_theta': has_theta,
            'has_dp': has_dp,
            'gt': gt,
            'dataset': '3dpw'
        }


if __name__ == '__main__':
    data = PW3D('D:/paper/human_body_reconstruction/datasets/human_reconstruction/3DPW/',
                split='train',
                image_scale_range=(0.2, 1.11),
                trans_scale=0.6,
                flip_prob=0.5,
                rot_prob=-1,
                rot_degree=45,
                box_stretch=28,
                keep_truncation_kps=True,
                min_truncation_kps_in_image=8,
                min_truncation_kps=12,
                min_vis_kps=6,
                max_data_len=-1)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)

    for batch in data_loader:

        debugger = Debugger(opt.smpl_path)
        img = batch['input'][0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((img + 1) / 2 * 255.), 0, 255).astype(np.uint8)

        # gt heat map
        gt_box_hm = debugger.gen_colormap(batch['box_hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, gt_box_hm, 'gt_box_hm')


        # decode bbox, key points
        decode_id = 'decode'
        debugger.add_img(img, img_id=decode_id)
        bbox = decode_label_bbox(batch['box_mask'][0], batch['box_ind'][0], batch['box_cd'][0], batch['box_wh'][0])
        kp2d = decode_label_kp2d(batch['kp2d_mask'][0], batch['kp2d'][0])
        for box in bbox:
            debugger.add_bbox(box, img_id=decode_id)
        for kp in kp2d:
            debugger.add_kp2d(kp, img_id=decode_id)


        # gt bbox, key points
        gt_id = 'gt_bbox_kp2d'
        debugger.add_img(img, img_id=gt_id)
        for obj in batch['gt']:
            debugger.add_bbox(obj['bbox'][0], img_id=gt_id)
            debugger.add_kp2d(obj['kp2d'][0], img_id=gt_id)


        # gt smpl
        gt_id = 'smpl'
        debugger.add_img(img, img_id=gt_id)
        for obj in batch['gt']:
            camera = get_camera_from_batch(obj['bbox'][0], opt.camera_pose_z)
            debugger.add_smpl(obj['pose'][0], obj['shape'][0], camera=camera, img_id=gt_id)


        debugger.show_all_imgs(pause=True)