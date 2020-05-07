import os
import time
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from torch.utils.data import DataLoader, ConcatDataset

from  utils.opts import opt

from .coco2017 import COCO2017
# from .coco2014 import COCO2014
from .lsp import Lsp
# from .lsp_ext import LspExt
from .hum36m import Hum36m



def coco_data_loader():
    datasets = []
    for name in opt.coco_data_set:
        path = opt.data_set_path[name]
        if name == 'coco2014':
            dataset = COCO2014(path)
        elif name == 'coco2017':
            dataset = COCO2017(
                data_path=path,
                split='train',
                image_scale_range=(0.6, 1.2),
                trans_scale=0.5,
                flip_prob=-1,
                rot_prob=-1,
                rot_degree=10,
                max_data_len=-1
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
        num_workers=opt.num_worker
    )


def lsp_data_loader():
    datasets = []
    for name in opt.lsp_data_set:
        path = opt.data_set_path[name]
        if name == 'lsp':
            dataset = Lsp(
                data_path=path,
                split='train',
                image_scale_range=(1, 1.01),
                trans_scale=0,
                flip_prob=-1,
                rot_prob=0.5,
                rot_degree=45,
                box_stretch=15,
                max_data_len=50
            )
        elif name == 'lsp_ext':
            dataset = LspExt(path)
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
        num_workers=opt.num_worker
    )


def hum36m_data_loader():
    datasets = []
    for name in opt.hum36m_data_set:
        path = opt.data_set_path[name]
        if name == 'hum36m':
            dataset = Hum36m(
                data_path=path,
                split='train',
                image_scale_range=(1.0, 1.01),
                trans_scale=0,
                flip_prob=1,
                rot_prob=0.5,
                rot_degree=45,
                box_stretch=20,
                max_data_len=50
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
        num_workers=opt.num_worker
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

        return data_list

