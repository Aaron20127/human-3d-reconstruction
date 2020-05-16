import os
import sys
import h5py
import numpy as np
import torch
import copy

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


file_dir = 'D:\paper\human_body_reconstruction\datasets\human_reconstruction\hum36m-toy'
src_file = os.path.join(file_dir, 'annot_cocoplus_19_3dkp.h5')
train_file = os.path.join(file_dir, 'train.h5')
val_file = os.path.join(file_dir, 'val.h5')

f_train =  h5py.File(train_file, 'w')
f_val =  h5py.File(val_file, 'w')


with h5py.File(src_file) as fp:
    kp2ds = np.array(fp['gt2d']).reshape(-1, 14, 3)
    kp3ds = np.array(fp['gt3d']).reshape(-1, 19, 3)
    shape = np.array(fp['shape'])
    pose = np.array(fp['pose'])
    imagename = np.array(fp['imagename'])


    train_kp2ds = np.zeros((kp2ds.shape[0], 14, 3))
    train_kp3ds = np.zeros((kp3ds.shape[0], 19, 3))
    train_shape = np.zeros((shape.shape[0], 10))
    train_pose = np.zeros((pose.shape[0], 72))
    train_imagename = copy.deepcopy(imagename)

    val_kp2ds = np.zeros((kp2ds.shape[0], 14, 3))
    val_kp3ds = np.zeros((kp3ds.shape[0], 19, 3))
    val_shape = np.zeros((shape.shape[0], 10))
    val_pose = np.zeros((pose.shape[0], 72))
    val_imagename = copy.deepcopy(imagename)

    num_train = 0
    num_val = 0
    for i in range(pose.shape[0]):
        if (i + 1) % 180 == 0:
            val_kp2ds[num_val] = kp2ds[i]
            val_kp3ds[num_val] = kp3ds[i]
            val_shape[num_val] = shape[i]
            val_pose[num_val] = pose[i]
            val_imagename[num_val] = imagename[i]

            num_val = num_val + 1
            print('val {}'.format(num_val))
        else:
            train_kp2ds[num_train] = kp2ds[i]
            train_kp3ds[num_train] = kp3ds[i]
            train_shape[num_train] = shape[i]
            train_pose[num_train] = pose[i]
            train_imagename[num_train] = imagename[i]

            num_train = num_train + 1
            print('train {}'.format(num_train))


f_train.create_dataset('gt2d', data=train_kp2ds[:num_train])
f_train.create_dataset('gt3d', data=train_kp3ds[:num_train])
f_train.create_dataset('shape', data=train_shape[:num_train])
f_train.create_dataset('pose', data=train_pose[:num_train])
f_train.create_dataset('imagename', data=train_imagename[:num_train])

f_val.create_dataset('gt2d', data=val_kp2ds[:num_val])
f_val.create_dataset('gt3d', data=val_kp3ds[:num_val])
f_val.create_dataset('shape', data=val_shape[:num_val])
f_val.create_dataset('pose', data=val_pose[:num_val])
f_val.create_dataset('imagename', data=val_imagename[:num_val])


print('total val {}'.format(num_val))
print('total trian {}'.format(num_train))
print('done.')