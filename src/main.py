
import os
import sys
import torch
import torch.utils.data


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/lib')

from opts import opts
from logger import Logger
from emailSender import send_email

from trains.trainer import HMRTrainer


def main(opt):
    ## 1. basic
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    opt.logger = Logger(opt)

    ## 2. create model
    trainer = HMRTrainer(opt)

    if opt.test:
        trainer.val()
    else:
        trainer.train()


    ## 3. send email
    send_email(opt)

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)