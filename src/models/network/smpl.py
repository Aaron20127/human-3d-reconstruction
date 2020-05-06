'''
    file:   SMPL.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from original SMPL
'''

import torch
import pickle
import sys
import os
import numpy as np
import torch.nn as nn
import cv2

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')
sys.path.insert(0, abspath + '/../../')

from model_util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose
from utils.render import rotation_x, weak_perspective, weak_perspective_render_obj

class SMPL(nn.Module):
    def __init__(self, model_path,
                       weight_batch_size=32*1,
                       joint_type='cocoplus'):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'rb') as f:
            # model = json.load(reader)
            model = pickle.load(f, encoding='iso-8859-1')

        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'].toarray(), dtype=np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_joint_regressor = np.array(model['cocoplus_regressor'].toarray(), dtype=np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        # weight_batch_size = max(args.batch_2d_size + args.batch_3d_size, args.eval_batch_size)
        np_weights = np.tile(np_weights, (weight_batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())


        # rotate x axis
        sin = np.sin(np.pi)
        cos = np.cos(np.pi)
        np_Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos, sin],
            [0.0, -sin, cos],
        ])
        self.register_buffer('Rx', torch.from_numpy(np_Rx).float())

        self.cur_device = None


    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, beta, theta):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor.T)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor.T)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor.T)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)  # 减去对角线元素
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor.T)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor.T)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor.T)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        # # rotate x
        # rx_verts = torch.matmul(verts, self.Rx.T)
        # rx_joints = torch.matmul(joints, self.Rx.T)

        return verts, joints, Rs, self.faces



if __name__ == '__main__':
    device = torch.device('cuda', 0)
    smpl = SMPL("D:/paper/human_body_reconstruction/code/master/data/neutral_smpl_with_cocoplus_reg.pkl",).to(device)


    ###1. get pose shape
    pose = (np.random.rand(24,3) - 0.5) * 0.4
    beta = (np.random.rand(10) - 0.5) * 0.6
    vbeta = torch.tensor(np.array([beta])).float().to(device)
    vpose = torch.tensor(np.array([pose])).float().to(device)

    ## get vertices and joints
    verts, joints, r, faces = smpl(vbeta, vpose)

    ## render
    camera = torch.tensor([1, 0, 0]).to(device)  # 弱透视投影参数s,cx,cy
    verts = weak_perspective(verts[0], camera).detach().cpu().numpy() # 对x,y弱透视投影，投影，平移，放缩
    J = weak_perspective(joints[0], camera).detach().cpu().numpy()
    obj = {
        'verts': verts,  # 模型顶点
        'faces': faces,  # 面片序号
        'J': J,  # 3D关节点
    }
    color_origin, depth = weak_perspective_render_obj(obj, width=512, height=512, show_smpl_joints=True)


    ### 2. reflect pose
    rpose = reflect_pose(pose)
    vpose = torch.tensor(np.array([rpose])).float().to(device)

    ## get vertices and joints
    verts, joints, r, faces = smpl(vbeta, vpose)

    ## render
    verts = weak_perspective(verts[0], camera).detach().cpu().numpy() # 对x,y弱透视投影，投影，平移，放缩
    J = weak_perspective(joints[0], camera).detach().cpu().numpy()
    obj = {
        'verts': verts,  # 模型顶点
        'faces': faces,  # 面片序号
        'J': J,  # 3D关节点
    }
    color_reflect, depth = weak_perspective_render_obj(obj, width=512, height=512, show_smpl_joints=True)

    # show
    cv2.imshow('origin', color_origin)
    cv2.imshow('reflect', color_reflect)
    cv2.waitKey(0)

    # rpose = reflect_pose(pose)
    # vpose = torch.tensor(np.array([rpose])).float().to(device)
    #
    # verts, j, r = smpl(vbeta, vpose, get_skin=True)
    # smpl.save_obj(verts[0].cpu().numpy(), './rmesh.obj')

