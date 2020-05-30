import os
import time
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch.utils.data import DataLoader, ConcatDataset

from  utils.opts import opt

from .coco2017 import COCO2017
from .coco2014 import COCO2014
from .lsp import Lsp
from .lsp_ext import LspExt
from .hum36m import Hum36m
from .pw3d import PW3D



def coco_data_loader():
    datasets = []
    for name in opt.coco_data_set:
        path = opt.data_set_path[name]
        if name == 'coco2014':
            dataset = COCO2014(
                data_path=path,
                split='train',
                image_scale_range=(0.4, 1.11),
                trans_scale=0.5,
                flip_prob=0.5,
                rot_prob=-1,
                rot_degree=30,
                min_vis_kps= opt.min_vis_kps,
                load_min_vis_kps=opt.load_min_vis_kps,
                max_data_len=-1,
                keep_truncation_kps = opt.keep_truncation_kps,
                min_truncation_kps_in_image=opt.min_truncation_kps_in_image,
                min_truncation_kps=opt.min_truncation_kps
            )
        elif name == 'coco2017':
            dataset = COCO2017(
                data_path=path,
                split='train',
                image_scale_range=(0.4, 1.11),
                trans_scale=0.5,
                flip_prob=0.5,
                rot_prob=-1,
                rot_degree=30,
                min_vis_kps= opt.min_vis_kps,
                load_min_vis_kps=opt.load_min_vis_kps,
                max_data_len=-1,
                keep_truncation_kps=opt.keep_truncation_kps,
                min_truncation_kps_in_image=opt.min_truncation_kps_in_image,
                min_truncation_kps=opt.min_truncation_kps
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.batch_size_coco,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opt.num_workers
    )


def lsp_data_loader():
    datasets = []
    for name in opt.lsp_data_set:
        path = opt.data_set_path[name]
        if name == 'lsp':
            dataset = Lsp(
                data_path=path,
                split='train',
                image_scale_range=(0.3, 1.01),
                trans_scale=0.5,
                flip_prob=0.5,
                rot_prob=0.5,
                rot_degree=20,
                box_stretch=30,
                max_data_len=-1,
                keep_truncation_kps=opt.keep_truncation_kps,
                min_truncation_kps_in_image=opt.min_truncation_kps_in_image,
                min_truncation_kps=opt.min_truncation_kps
            )
        elif name == 'lsp_ext':
            dataset = LspExt(
                data_path=path,
                split='train',
                image_scale_range=(0.3, 1.01),
                trans_scale=0.5,
                flip_prob=0.5,
                rot_prob=0.5,
                rot_degree=20,
                box_stretch=30,
                max_data_len=-1,
                keep_truncation_kps = opt.keep_truncation_kps,
                min_truncation_kps_in_image=opt.min_truncation_kps_in_image,
                min_truncation_kps=opt.min_truncation_kps
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.batch_size_lsp,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opt.num_workers
    )


def hum36m_data_loader():
    datasets = []
    for name in opt.hum36m_data_set:
        path = opt.data_set_path[name]
        if name == 'hum36m':
            dataset = Hum36m(
                data_path=path,
                split='train',
                image_scale_range=(0.4, 1.11),
                trans_scale=0.5,
                flip_prob=0.5,
                rot_prob=opt.hum36m_rot_prob,
                rot_degree=opt.hum36m_rot_degree,
                box_stretch=20,
                max_data_len=-1,
                keep_truncation_kps = opt.keep_truncation_kps,
                min_truncation_kps_in_image = opt.min_truncation_kps_in_image,
                min_truncation_kps = opt.min_truncation_kps
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.batch_size_hum36m,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opt.num_workers
    )


def pw3d_data_loader():
    datasets = []
    for name in opt.pw3d_data_set:
        path = opt.data_set_path[name]
        if name == '3dpw':
            dataset = PW3D(
                data_path=path,
                split='train',
                image_scale_range=(0.2, 1.11),
                trans_scale=0.6,
                flip_prob=0.5,
                rot_prob=opt.pw3d_rot_prob,
                rot_degree=opt.pw3d_rot_degree,
                box_stretch=28,
                max_data_len=-1,
                keep_truncation_kps=opt.keep_truncation_kps,
                min_truncation_kps_in_image=opt.min_truncation_kps_in_image,
                min_truncation_kps=opt.min_truncation_kps
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.batch_size_3dpw,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opt.num_workers
    )


def val_coco_data_loader():
    datasets = []
    for name in opt.coco_val_data_set:
        path = opt.data_set_path[name]
        if name == 'coco2014':
            dataset = COCO2014(
                data_path=path,
                split='val',
                image_scale_range=(1.0, 1.01),
                trans_scale=0,
                flip_prob=-1,
                rot_prob=-1,
                rot_degree=30,
                min_vis_kps=0,
                load_min_vis_kps=6,
                max_data_len=-1
            )
        elif name == 'coco2017':
            dataset = COCO2017(
                data_path=path,
                split='val',
                image_scale_range=(1.0, 1.01),
                trans_scale=0,
                flip_prob=-1,
                rot_prob=-1,
                rot_degree=30,
                min_vis_kps=0,
                load_min_vis_kps=6,
                max_data_len=-1
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.val_batch_size_coco,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=opt.num_workers
    )

def val_hum36m_data_loader():
    datasets = []
    for name in opt.hum36m_val_data_set:
        path = opt.data_set_path[name]
        if name == 'hum36m':
            dataset = Hum36m(
                data_path=path,
                split='val',
                image_scale_range=(1.0, 1.01),
                trans_scale=0,
                flip_prob=-1,
                rot_prob=-1,
                rot_degree=45,
                box_stretch=20,
                max_data_len=-1
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.val_batch_size_hum36m,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=opt.num_workers
    )


def val_3dpw_data_loader():
    datasets = []
    for name in opt.pw3d_val_data_set:
        path = opt.data_set_path[name]
        if name == '3dpw':
            dataset = PW3D(
                data_path=path,
                split='val',
                image_scale_range=(1.0, 1.01),
                trans_scale=0,
                flip_prob=-1,
                rot_prob=-1,
                rot_degree=45,
                box_stretch=28,
                max_data_len=-1
            )
        else:
            msg = 'invalid dataset {}.'.format(name)
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opt.val_batch_size_3dpw,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=opt.num_workers
    )


class multi_data_loader(object):
    """
    Asseble multipule DataLoaders to one DataLoader.

    Arguments:
        dataloaders (list): lists of DataLoader.
    """

    def __init__(self, dataloaders):
        self.loaders = dataloaders
        self.iter_loaders = \
            [iter(dl) for dl in dataloaders]

    def __len__(self):
        """ return max length of DataLoader. """
        length = 0
        for dl in self.loaders:
            length = max(length, len(dl))
        return length

    def __next__(self):
        return self.next()

    def next(self):
        """ return all of data in all of DataLoader. """
        data_list = []

        for i in range(len(self.iter_loaders)):
            try:
                data = next(self.iter_loaders[i])
            except StopIteration:
                self.iter_loaders[i] = iter(self.loaders[i])
                data = next(self.iter_loaders[i])

            data_list.append(data)

        batch = self.merge_batch(data_list)
        return batch

    def merge_batch(self, batch):
        try:
            keys = self.batch_keys
        except:
            self.store_batch_keys(batch)
            keys = self.batch_keys

        label = {}
        gt = {}
        for k in keys:
            st = []
            for b in batch:
                if k in b: st.append(b[k])

            if type(st[0]) == torch.Tensor:
                label[k] = torch.cat(st, 0)
            else:
                gt[k] = st

        return {
            'label': label,
            'gt': gt
        }

    def store_batch_keys(self, batch):
        batch_keys = []
        for b in batch:
            for k in b.keys():
                if k not in batch_keys:
                    batch_keys.append(k)

        self.batch_keys = batch_keys

