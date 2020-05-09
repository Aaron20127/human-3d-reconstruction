import os
import sys
import h5py
import numpy as np
import torch
import time
import pycocotools.coco as coco

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')




file_dir = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017/annotations'
# src_file = os.path.join(file_dir, 'person_keypoints_train2017.json')
# dst_file = os.path.join(file_dir, 'person_keypoints_train2017.h5')
src_file = os.path.join(file_dir, 'person_keypoints_val2017.json')
dst_file = os.path.join(file_dir, 'person_keypoints_val2017.h5')

t1 = time.time()
with h5py.File(dst_file, 'w') as dst_f:
    coco = coco.COCO(src_file)
    image_ids = coco.getImgIds()

    # person and not crowd
    images = []
    annotations = []


    for img_id in image_ids:  # only save the image ids who have annotations
        idxs = coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
        anns = coco.loadAnns(ids=idxs)
        if len(anns) > 0:
            images.append(img_id)

            obj = []
            for ann in anns:
                obj.append({
                    'bbox': ann['bbox'],
                    'keypoints': ann['keypoints'],
                    'num_keypoints': ann['num_keypoints']
                })

            annotations.append(obj)

    dst_f.create_dataset('images', data=np.array(images))
    dst_f.create_dataset('anns', data=np.array(annotations))

t2 = time.time()

print('done. t:({})'.format(t2-t1))




