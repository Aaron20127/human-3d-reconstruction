import os
import sys
import h5py
import numpy as np
import torch

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from models.network.smpl import SMPL


file_dir = 'D:\paper\human_body_reconstruction\datasets\human_reconstruction\hum36m-toy'
src_file = os.path.join(file_dir, 'annot.h5')
dst_file = os.path.join(file_dir, 'annot_val.h5')


with h5py.File(dst_file, 'w') as dst_f:
    with h5py.File(src_file) as fp:
        kp2ds = np.array(fp['gt2d']).reshape(-1, 14, 3)
        # self.kp3ds = np.array(fp['gt3d']).reshape(-1, 14, 3)
        shape = np.array(fp['shape'])
        pose = np.array(fp['pose'])
        imagename = np.array(fp['imagename'])

        smpl = SMPL(
            "D:/paper/human_body_reconstruction/code/master/data/neutral_smpl_with_cocoplus_reg.pkl").cuda()

        kp3ds = np.zeros((pose.shape[0], 19, 3))
        for i in range(pose.shape[0]):
            p = torch.tensor(pose[i], dtype=torch.float32).reshape(24,3).cuda()
            s = torch.tensor(shape[i], dtype=torch.float32).reshape(1,10).cuda()

            verts, joints, r, faces = smpl(s, p)
            kp3ds[i] = np.array(joints[0].cpu())
            print(kp3ds[i], pose[i].reshape(24,3), shape[i])

            print('{} / {}'.format(i, pose.shape[0]))

    dst_f.create_dataset('gt2d', data=kp2ds)
    dst_f.create_dataset('gt3d', data=kp3ds)
    dst_f.create_dataset('shape', data=shape)
    dst_f.create_dataset('pose', data=pose)
    dst_f.create_dataset('imagename', data=imagename)

print('done.')