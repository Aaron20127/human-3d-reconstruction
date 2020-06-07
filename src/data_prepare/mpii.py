import os
import cv2
import numpy as np
import pickle
import math
import os
from os.path import join
from scipy.io import loadmat
import h5py


def mpii_extract(dataset_path, is_train=True):
    # structs we use
    MIN_P = 14
    imgnames_ = []
    kp2d_ = np.zeros((30000, 17, 16, 3))

    # annotation files
    annot_file = os.path.join(dataset_path, 'mpii_human_pose_v1_u12_1.mat')

    # read annotations
    # f = h5py.File(annot_file, 'r')
    res = loadmat(annot_file,struct_as_record=False, squeeze_me=True)
    f = res['RELEASE']
    all_ids = np.array(range(len(f.annolist)))

    if is_train:
        img_inds = all_ids[f.img_train.astype('bool')]
    else:
        img_inds = all_ids[np.logical_not(f.img_train)]

    k = 0
    for id,img_id in enumerate(img_inds):
        anno_info=f.annolist[img_id]
        image_name=anno_info.image.name
        kp2ds=anno_info.annorect
        joints = np.zeros((17, 16, 3))

        print('{} / {}'.format(id, len(img_inds)))

        img = cv2.imread(dataset_path + '/images/' + image_name)

        try:
            h, w, _ = img.shape
        except:
            continue

        try:
            kp2ds_shape=kp2ds.shape[0]
        except AttributeError:
            kp2ds_shape = 1

        try:
            single_person=f.single_person[img_id]
            if not isinstance(single_person, np.ndarray):
                single_person = np.array([single_person])

            if single_person.shape[0]<1:
                continue

            elif kp2ds_shape == 1:   #  only one person
                kp = kp2ds.annopoints.point
                joint = np.zeros((16,3))

                try:
                    kp_shape = kp.shape[0]
                except AttributeError:
                    kp_shape = 0

                if kp_shape < MIN_P:
                    continue

                for j in range(kp.shape[0]):
                    p = kp[j]
                    joint[p.id, 0] = p.x
                    joint[p.id, 1] = p.y
                    if isinstance(p.is_visible, int) or isinstance(p.is_visible, str):
                        joint[p.id, 2] = int(p.is_visible)
                    else:
                        joint[p.id, 2] = 1

                if joint[:,2].sum() < MIN_P:
                    continue

                if np.sum((joint[:, 0] < 0).astype(np.float32) + (joint[:, 0] > w).astype(np.float32) + \
                          (joint[:, 1] < 0).astype(np.float32) + (joint[:, 1] > h).astype(np.float32)) > 0:
                    continue

                joints[0] = joint

                # img_path = join(dataset_path, 'images', image_name)
                #visiual_Image(img_path, joint)
                #print(img_path)
            elif kp2ds_shape > 1 :
                # if max_kp2ds_shape < kp2ds_shape:
                #     max_kp2ds_shape = kp2ds_shape
                # print(max_kp2ds_shape)
                for i in single_person:
                    kp=kp2ds[i-1].annopoints.point
                    joint=np.zeros((16, 3))

                    try:
                        kp_shape = kp.shape[0]
                    except AttributeError:
                        kp_shape = 0

                    if kp_shape < MIN_P:
                        continue

                    for j in range(kp.shape[0]):
                        p=kp[j]
                        joint[p.id,0]=p.x
                        joint[p.id,1]=p.y
                        if isinstance(p.is_visible, int) or isinstance(p.is_visible, str):
                            joint[p.id, 2] = int(p.is_visible)
                        else:
                            joint[p.id,2]=1

                    if joint[:,2].sum() < MIN_P:
                        continue

                    if np.sum((joint[:, 0] < 0).astype(np.float32) + (joint[:, 0] > w).astype(np.float32) + \
                              (joint[:, 1] < 0).astype(np.float32) + (joint[:, 1] > h).astype(np.float32)) > 0:
                        continue

                    joints[i-1] = joint
                    # img_path=join(dataset_path,'images',image_name)
                    #visiual_Image(img_path,joint)
                    #print(img_path)

        except  AttributeError:
            continue

        if joints[:,:,2].sum() <= 0:
            continue

        imgnames_.append(os.path.join('images', image_name))
        kp2d_[k] = joints
        k +=1

    return {
        'imagename': imgnames_,
        'kp2d': kp2d_[:k]
    }

    """
    # go over all annotated examples
    for center, imgname, part16, scale in zip(centers, imgnames, parts, scales):
        imgname = imgname.decode('utf-8')
        # check if all major body joints are annotated
        if (part16 > 0).sum() < 2 * len(joints_idx):
            continue
        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = np.hstack([part16, np.ones([16, 1])])
        # read openpose detections
        # json_file = os.path.join(openpose_path, 'mpii',
        #    imgname.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'mpii')

        # store data
        imgnames_.append(join('images', imgname))
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        # openposes_.append(openpose)
    """


    # # store the data struct
    # if not os.path.isdir(out_path):
    #     os.makedirs(out_path)
    #
    # if is_train:
    #     out_file = os.path.join(out_path, 'mpii_train.npz')
    # else:
    #     out_file = os.path.join(out_path, 'mpii_test.npz')
    #
    # np.savez(out_file, imgname=imgnames_,
    #          center=centers_,
    #          scale=scales_,
    #          part=parts_,)


if __name__=="__main__":
    data_path = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/mpii'
    data = mpii_extract(data_path, True)

    dst_file = data_path + '/train.h5'

    with h5py.File(dst_file, 'w') as fp:
        fp.create_dataset('kp2d', data=data['kp2d'])
        fp.create_dataset('imagename', data=np.array(data['imagename'], dtype='S'))

    print('total trian {}'.format(len(data['kp2d'])))
    print('done.')