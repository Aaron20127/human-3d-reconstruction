
import os
import sys
import torch
import torch.utils.data


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


from utils.opts import opt
from utils.util import pre_process
from utils.logger import Logger
from utils.emailSender import send_email
from train.trainer import HMRTrainer

def main(opt):
    ## 1. basic
    pre_process(opt)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    opt.device = torch.device('cuda' if -1 not in opt.gpus_list else 'cpu')
    opt.logger = Logger(opt)

    ## 2. create model
    trainer = HMRTrainer(opt)

    if opt.val:
        trainer.val()
    else:
        trainer.train()

    #### 3. send email
    # send_email(opt)

if __name__ == '__main__':
    main(opt)