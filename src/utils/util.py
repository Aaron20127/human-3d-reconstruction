import numpy as np
import cv2
import random
import copy
import time
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


class Clock:
    """ timer """
    def __init__(self):
        self.start_time = time.time()
        self.pre_time = self.start_time

    def update(self):
        """ update initial value elapsed time """
        self.pre_time = time.time()

    def elapsed(self):
        """ compute the time difference from the last call. """
        cur_time = time.time()
        elapsed = cur_time - self.pre_time
        self.pre_time = cur_time
        return elapsed

    def total(self):
        """ calculate the time from startup to now. """
        total = time.time() - self.start_time
        return total


def str_time(seconds):
    """ format seconds to h:m:s. """
    H = int(seconds / 3600)
    M = int((seconds - H * 3600) / 60)
    S = int(seconds - H * 3600 - M * 60)
    H = str(H) if H > 9 else '0' + str(H)
    M = str(M) if M > 9 else '0' + str(M)
    S = str(S) if S > 9 else '0' + str(S)
    return '{}:{}:{}'.format(H, M, S)


def show_net_para(net):
    """ calculate parameters of network """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total: %d , trainable: %d' % (total_num, trainable_num))


""" rotation """
def Rx_mat(theta):
    """绕x轴旋转
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])

def Ry_mat(theta):
    """绕y轴旋转
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

def Rz_mat(theta):
    """绕z轴旋转
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


'''
    purpose:
        reflect poses, when the image is reflect by left-right

    Argument:
        poses (array, 72): 72 real number
'''
def reflect_pose(poses):
    swap_inds = np.array([
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
        19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
        36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
        50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
        67, 68
    ])

    sign_flip = np.array([
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
        -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1
    ])

    return poses[swap_inds] * sign_flip