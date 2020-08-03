
import time
import os
abspath = os.path.abspath(os.path.dirname(__file__))


import argparse
parser = argparse.ArgumentParser()
agt = parser.add_argument

agt('--experiment_name',  default='network_width')
agt('--network', default='BaseNet_12')

agt('--weight_decay',  default=0, type=float)
agt('--kp2d_weight',  default=10, type=float)
agt('--lr',  default=1e-4, type=float)
agt('--epoches',  default=20000, type=int)
agt('--num_workers',  default=2, type=int)
agt('--val', action='store_true')
agt('--batch_size', default=128, type=int)
agt('--gpus', default='1', help='-1 for CPU, use comma for multiple gpus')
# agt('--crowdpose_path', default='F:\\paper\\dataset\\crowdpose')
agt('--crowdpose_path', default='/home/icvhpc1/bluce/dataset/crowdpose')
# agt('--load_model', default='/home/icvhpc1/bluce/code/mhmr-v1.1/test/experiment_2/refer_kp2d_from_other_kp2d/log/BaseNet_10/2020-07-28_11-41-29/best_model.pth')
agt('--load_model', default='')



opt = parser.parse_args()

# preprocess
opt.gpus_list = [int(i) for i in opt.gpus.split(',')]
opt.device = 'cuda' if -1 not in opt.gpus_list else 'cpu'

log_id = time.strftime('%Y-%m-%d_%H-%M-%S')
opt.log_dir = abspath + '/log/' +  opt.experiment_name + '/' + opt.network + '/' + 'weight_decay_' + str(opt.weight_decay) + '/' + log_id + '/'