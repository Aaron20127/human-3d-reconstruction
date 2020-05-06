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
    return np.mat([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])

def Ry_mat(theta):
    """绕y轴旋转
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.mat([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

def Rz_mat(theta):
    """绕z轴旋转
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.mat([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])