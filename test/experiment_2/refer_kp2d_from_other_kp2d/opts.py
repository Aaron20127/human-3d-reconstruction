import argparse
parser = argparse.ArgumentParser()
agt = parser.add_argument


agt('--kp2d_weight',  default=10, type=float)
agt('--lr',  default=1e-4, type=float)
agt('--epoches',  default=20000, type=int)
agt('--val', action='store_true')
agt('--batch_size', default=128, type=int)


opt = parser.parse_args()