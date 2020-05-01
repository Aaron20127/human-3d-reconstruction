import os
import time
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from torch.utils.data import DataLoader, ConcatDataset

from coco2017_dataset import COCO2017
from coco2014_dataset import COCO2014
from lsp_dataset import Lsp
from lsp_ext_dataset import LspExt
from hum36m_dataset import Hum36m

from  utils.opts import opts


def lsp_data_loader():
    datasets = []
    for data_set_name in opts.coco_data_set:
        data_set_path = opts.data_set_path[data_set_name]
        if data_set_name == 'coco2014':
            dataset = COCO2014(data_set_path)
        elif data_set_name == 'coco2017':
            dataset = COCO2017(data_set_path)
        else:
            msg = 'invalid dataset'
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opts.batch_size_coco,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opts.num_worker
    )


def lsp_data_loader():
    datasets = []
    for data_set_name in opts.lsp_data_set:
        data_set_path = opts.data_set_path[data_set_name]
        if data_set_name == 'lsp':
            dataset = Lsp(data_set_path)
        elif data_set_name == 'lsp_ext':
            dataset = LspExt(data_set_path)
        else:
            msg = 'invalid dataset.'
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opts.batch_size_lsp,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opts.num_worker
    )


def hum36m_data_loader():
    datasets = []
    for data_set_name in opts.hum36m_data_set:
        data_set_path = opts.data_set_path[data_set_name]
        if data_set_name == 'hum36m':
            dataset = Hum36m(data_set_path)
        else:
            msg = 'invalid dataset.'
            sys.exit(msg)

        datasets.append(dataset)

    new_datasets = ConcatDataset(datasets)

    return DataLoader(
        dataset=new_datasets,
        batch_size=opts.batch_size_hum36m,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=opts.num_worker
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

