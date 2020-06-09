
# from .opts import opt
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def pre_process(opt):
    """log"""
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    exp_dir = os.path.join(root_dir, 'exp')
    opt.save_dir = os.path.join(exp_dir, opt.task, opt.subtask)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    opt.debug_image_dir = os.path.join(opt.debug_dir, 'images')
    opt.debug_obj_dir = os.path.join(opt.debug_dir, 'objs')
    opt.log_dir = opt.save_dir + '/logs'
    opt.pr_curve_dir =  opt.log_dir + '/pr_curve'

    """train"""
    opt.iou_thresh = [float(i) for i in opt.iou_thresh.split(',')]
    opt.gpus_list = [int(i) for i in opt.gpus.split(',')]
    opt.coco_data_set=[i for i in opt.coco_data_set.split(',')]
    opt.lsp_data_set=[i for i in opt.lsp_data_set.split(',')]
    opt.mpii_data_set=[i for i in opt.mpii_data_set.split(',')]
    opt.hum36m_data_set=[i for i in opt.hum36m_data_set.split(',')]
    opt.pw3d_data_set=[i for i in opt.pw3d_data_set.split(',')]
    opt.coco_val_data_set=[i for i in opt.coco_val_data_set.split(',')]
    opt.hum36m_val_data_set=[i for i in opt.hum36m_val_data_set.split(',')]
    opt.pw3d_val_data_set=[i for i in opt.pw3d_val_data_set.split(',')]
    opt.kp2d_every_weight_train=[float(i) for i in opt.kp2d_every_weight_train.split(',')]
    opt.kp2d_every_weight_val=[float(i) for i in opt.kp2d_every_weight_val.split(',')]

    if opt.hum36m_rot_prob or opt.pw3d_rot_prob > 0:
        opt.kp3d_weight = 0

    """model"""
    if opt.resume and opt.load_model == '':
        opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')

    """debug"""
    if opt.debug > 0:
        if opt.batch_size_coco + opt.batch_size_lsp + opt.batch_size_hum36m > 1:
            assert 0, 'needed one batch size to debug.'

        if opt.val_batch_size_coco + opt.val_batch_size_hum36m > 1:
            assert 0, 'needed one batch size to debug.'


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
        batch x theta
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    M = torch.zeros((theta.size(0), 3, 3), requires_grad=False).to(theta.device)
    M[:, 0, 0]=1
    M[:, 1, 1]=cos
    M[:, 1, 2]=-sin
    M[:, 2, 1]=sin
    M[:, 2, 2]=cos

    return M


def Ry_mat(theta):
    """绕y轴旋转
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    M = torch.zeros((theta.size(0), 3, 3), requires_grad=False).to(theta.device)
    M[:, 1, 1]=1
    M[:, 0, 0]=cos
    M[:, 0, 2]=sin
    M[:, 2, 0]=-sin
    M[:, 2, 2]=cos

    return M


def Rz_mat(theta):
    """绕z轴旋转
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    M = torch.zeros((theta.size(0), 3, 3), requires_grad=False).to(theta.device)

    M[:, 2, 2]=1
    M[:, 0, 0]=cos
    M[:, 0, 1]=-sin
    M[:, 1, 0]=sin
    M[:, 1, 1]=cos

    return M


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
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1,
        1, -1, -1
    ])

    return poses[swap_inds] * sign_flip




###################### loss ##########################

def sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def gather_feat(feat, ind, mask=None):
    '''use index(ind) to get value from feature map '''
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


############################ smpl ###############################
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    # theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def batch_global_rigid_transformation(Rs, Js, parent, device, rotate_base=False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).to(device)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).to(device)], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


def batch_lrotmin(theta):
    theta = theta[:,3:].contiguous()
    Rs = batch_rodrigues(theta.view(-1, 3))
    print(Rs.shape)
    e = Variable(torch.eye(3).float())
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)


def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    return (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)


############################# decode ##############################
def decode_label_bbox(mask, ind, cd, wh, down_ratio=4.0, output_res=128):
    mask = np.array(mask)
    ind = np.array(ind)
    cd = np.array(cd)
    wh = np.array(wh)

    inds = ind[mask==1]
    cds =  cd[mask==1]
    whs =  wh[mask==1]

    center = (np.stack((inds % output_res, inds // output_res),1) + cds) * down_ratio
    l_t = center - whs / 2.0 * down_ratio
    r_b = center + whs / 2.0 * down_ratio

    bbox = np.hstack((l_t, r_b))

    return bbox


def decode_label_kp2d(mask, kp2d):
    mask = np.array(mask)
    kp2d = np.array(kp2d)

    kp2ds = kp2d[mask == 1]

    return kp2ds


def decode_label_densepose(mask, dp2d, dp_ind, dp_rat):
    mask = np.array(mask)
    dp2d = np.array(dp2d)
    dp_ind = np.array(dp_ind)
    dp_rat = np.array(dp_rat)

    dp2d = dp2d[mask == 1]
    dp_ind = dp_ind[mask == 1]
    dp_rat = dp_rat[mask == 1]

    return dp2d, dp_ind, dp_rat


def get_camera_from_batch(bbox, camera_pose_z):
    bbox = bbox.cpu().numpy()

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # fx = (w+h) / 2.0 * opt.camera_pose_z
    # fx = max(w,h) / 2. * opt.camera_pose_z
    fx = math.sqrt(w*h) * camera_pose_z
    fy = fx
    cx = (bbox[0] + bbox[2]) / 2.
    cy = (bbox[1] + bbox[3]) / 2.

    k = np.eye(3, 3)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy

    t = np.array([[0, 0, camera_pose_z]]).T

    return {
        'k': k,
        't': t
    }


def perspective_transform(kp3d, camera):
    kp3d_t = kp3d.T + camera['t']
    kp3d_h = np.dot(camera['k'], kp3d_t)
    kp2d = kp3d_h / kp3d_h[2, :]

    return kp2d.T

