
import os
import sys
import torch
import torch.utils.data
import numpy as np
import random

# OAM错误
os.environ['KMP_DUPLICATE_LIB_OK']='True'

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


from utils.opts import opt
from utils.logger import Logger
from utils.emailSender import send_email
from train.trainer import HMRTrainer

def main(opt):
    ## 1. basic
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

    opt.logger = Logger(opt)

    ## 2. create model
    trainer = HMRTrainer(opt)
    trainer.set_device(opt.gpus_list, opt.chunk_sizes, opt.device)

    if opt.val:
        trainer.val()
    else:
        trainer.train()

    #### 3. send email
    # send_email(opt)

if __name__ == '__main__':
    main(opt)