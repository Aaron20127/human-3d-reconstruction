import os
import sys
import pickle as pkl
import cv2
import pyrender
import trimesh
import h5py
import torch
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../")

import pickle
# from models.network.smpl import SMPL
# from utils.smpl_np import SMPL_np
import warnings
import numpy as np
warnings.simplefilter("always")


def get_bbox_from_kps(kp, img_size, box_stretch=100):
    w, h = img_size

    v_kp = kp[kp[:, 2] > 0]
    x_min = v_kp[:, 0].min()
    x_max = v_kp[:, 0].max()
    y_min = v_kp[:, 1].min()
    y_max = v_kp[:, 1].max()

    x_l = x_min - box_stretch if x_min - box_stretch > 0 else 0
    y_l = y_min - box_stretch if y_min - box_stretch > 0 else 0  # head special handle
    x_r = x_max + box_stretch if x_max + box_stretch < w - 1 else w - 1
    y_r = y_max + box_stretch if y_max + box_stretch < h - 1 else h - 1

    bbox = [x_l, y_l, x_r, y_r]

    return bbox



def crop_data(data_slot, label_path, images_path, save_path):

    save_image_dir = save_path + '/' + 'images'
    annotation_file = save_path + '/' + 'annotation.h5'
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)

    _kp2ds = np.zeros((24000, 18, 3))
    _shape = np.zeros((24000, 10))
    _pose = np.zeros((24000, 72))
    _imagename = []

    label_files = []
    for lable in data_slot:
        label_files.append(os.path.join(
            label_path, lable + '.pkl.h5'
        ))

    k = 0
    for file in label_files:
        with h5py.File(file, 'r') as fp:
            kp2ds = np.array(fp['gt2d'])
            shape = np.array(fp['shape'])
            pose = np.array(fp['pose'])
            imagename = np.array(fp['imagename'])

            for i in range(len(kp2ds)):
                print('{} / {}'.format(i, len(kp2ds)))
                img = cv2.imread(images_path + '/' + imagename[i].decode())

                for j in range(len(kp2ds[i])):
                    kp2d = kp2ds[i][j]

                    if np.sum(kp2d[:, 2]) == 0:
                        continue

                    _kp2ds[k] = kp2ds[i][j]
                    _shape[k] = shape[i][j]
                    _pose[k] = pose[i][j]

                    ## save crop image
                    bbox = get_bbox_from_kps(kp2d, (img.shape[1],img.shape[0]), box_stretch=80)
                    new_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    cv2.imwrite(save_image_dir + '/{}.jpg'.format(str(k).zfill(6)), new_img)

                    k = k + 1



    dst_fp = h5py.File(annotation_file, 'w')

    dst_fp.create_dataset('gt2d', data=_kp2ds[:k])
    # dst_fp.create_dataset('gt3d', data=_kp3ds[:k])
    dst_fp.create_dataset('shape', data=_shape[:k])
    dst_fp.create_dataset('pose', data=_pose[:k])
    dst_fp.create_dataset('imagename', data=np.array(_imagename[:k], dtype='S'))
    dst_fp.close()

    print('done, total {} '.format(k))



if __name__ == '__main__':
    data_slot = ['courtyard_dancing_00']
    label_path = 'F:\\paper\\dataset\\3DPW\\annotations\\validation'
    images_path = 'F:\\paper\\dataset\\3DPW'
    save_path = 'F:\\paper\\experiment_dataset\\experiment_1\\3dpw_crop'
    crop_data(data_slot, label_path, images_path, save_path)

