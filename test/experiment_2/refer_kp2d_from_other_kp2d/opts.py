import argparse
parser = argparse.ArgumentParser()
agt = parser.add_argument


agt('--kp2d_weight',  default=10, type=float)
agt('--lr',  default=1e-4, type=float)
agt('--epoches',  default=20000, type=int)
agt('--val', action='store_true')
agt('--batch_size', default=128, type=int)
agt('--gpus', default='-1', help='-1 for CPU, use comma for multiple gpus')
agt('--crowdpose_path', default='F:\\paper\\dataset\\crowdpose')



opt = parser.parse_args()

# preprocess
opt.gpus_list = [int(i) for i in opt.gpus.split(',')]
opt.device = 'cuda' if -1 not in opt.gpus_list else 'cpu'