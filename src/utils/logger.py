from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch

USE_TENSORBOARD = True
try:
   import tensorboardX
   print('Using tensorboard')
except:
   USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        ## make save dir
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.exists(opt.debug_dir):
            os.makedirs(opt.debug_dir)
   
        ## save options
        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('{}\n\n'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))

            opt_file.write('Note:\n')
            opt_file.write('    {}\n'.format(opt.note))

            opt_file.write('\nPytorch:\n')
            opt_file.write('    torch version: {}\n'.format(torch.__version__))
            opt_file.write('    cudnn version: {}\n'.format(torch.backends.cudnn.version()))

            opt_file.write('\nCmd:\n')
            opt_file.write(str(sys.argv))

            opt_file.write('\n\nOpt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('    %s: %s\n' % (str(k), str(v)))
          
        ## log file of train and val
        log_dir = opt.save_dir + '/logs'
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

        self.train_file = log_dir + '/train.txt'
        self.val_file = log_dir + '/val.txt'


    def write_file(self, file, text):
        """ write text to file """
        with open(file, 'a') as f:
            f.write(text)

    def write(self, split, text):
        """ write text  """
        if split =='train':
            self.write_file(self.train_file, text)
        elif split == 'val':
            self.write_file(self.val_file, text)
        else:
            assert 0, 'invalid log file.'

    def scalar_summary(self, tag, value, step):
        """ Log a scalar variable. """
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)

    def add_graph(self, model, input_to_model=None):
        """ add a graph. """
        if USE_TENSORBOARD:
            self.writer.add_graph(model, input_to_model)
