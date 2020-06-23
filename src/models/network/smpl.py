# '''
#     file:   SMPL.py
#
#     date:   2018_05_03
#     author: zhangxiong(1025679612@qq.com)
#     mark:   the algorithm is cited from original SMPL
# '''
#
# import torch
# import pickle
# import sys
# import os
# import numpy as np
# import torch.nn as nn
# import cv2
#
# abspath = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, abspath + '/../')
# sys.path.insert(0, abspath + '/../../')
#
# from utils.util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose
# from utils.render import perspective_render_obj_debug
#
#
# class SMPL(nn.Module):
#     def __init__(self, model_path,
#                        weight_batch_size=128*1,
#                        smpl_type='cocoplus'
#                        ):
#         super(SMPL, self).__init__()
#
#         if smpl_type not in ['cocoplus', 'basic']:
#             msg = 'unknow joint type: {}, it must be either "cocoplus" or "basic"'.format(smpl_type)
#             sys.exit(msg)
#
#
#         # if output_joint_type is not None and \
#         #         output_joint_type != smpl_type:
#         #     if output_joint_type == 'cocoplus' and smpl_type == 'basic':
#         #         self.kps_map = [8,5,2,1,4,7,21,19,17,16,18,20,12,0,0,0,0,0,0]  # map basic to cocoplus key points
#         #         self.not_exist_kps = [13, 14, 15, 16, 17, 18]
#
#
#         with open(model_path, 'rb') as f:
#             # model = json.load(reader)
#             model = pickle.load(f, encoding='iso-8859-1')
#
#         self.faces = model['f']
#
#         np_v_template = np.array(model['v_template'], dtype=np.float)
#         self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
#         self.size = [np_v_template.shape[0], 3]
#
#         np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
#         self.num_betas = np_shapedirs.shape[-1]
#         np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
#         self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())
#
#         np_J_regressor = np.array(model['J_regressor'].toarray(), dtype=np.float)
#         self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())
#
#         np_posedirs = np.array(model['posedirs'], dtype=np.float)
#         num_pose_basis = np_posedirs.shape[-1]
#         np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
#         self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())
#
#         self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
#
#         # np_joint_regressor = np.array(model['cocoplus_regressor'].toarray(), dtype=np.float)
#         # if smpl_type == 'basic':
#         #     self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
#         # else:
#         #     self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())
#
#         if smpl_type == 'basic':
#             np_joint_regressor = np.array(model['J_regressor'].toarray(), dtype=np.float)
#         elif smpl_type == 'cocoplus':
#             np_joint_regressor = np.array(model['cocoplus_regressor'].toarray(), dtype=np.float)
#
#         self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())
#
#
#         np_weights = np.array(model['weights'], dtype=np.float)
#         vertex_count = np_weights.shape[0]
#         vertex_component = np_weights.shape[1]
#
#         # weight_batch_size = max(args.batch_2d_size + args.batch_3d_size, args.eval_batch_size)
#         np_weights = np.tile(np_weights, (weight_batch_size, 1))
#         self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
#
#         self.register_buffer('e3', torch.eye(3).float())
#
#         self.cur_device = None
#
#
#     def save_obj(self, verts, obj_mesh_name):
#         # if not self.faces:
#         #     msg = 'obj not saveable!'
#         #     sys.exit(msg)
#
#         with open(obj_mesh_name, 'w') as fp:
#             for v in verts:
#                 fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#
#             for f in self.faces:  # Faces are 1-based, not 0-based in obj files
#                 fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
#
#
#     def forward(self, beta, theta):
#         device = beta.device
#         self.cur_device = torch.device(device.type, device.index)
#
#         num_batch = beta.shape[0]
#         # print('smpl num_batch {}'.format(num_batch))
#
#
#         v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
#         Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor.T)
#         Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor.T)
#         Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor.T)
#         J = torch.stack([Jx, Jy, Jz], dim=2)
#
#         Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
#         pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)  # 减去对角线元素
#         v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
#         self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=True)
#
#         weight = self.weight[:num_batch]
#         W = weight.view(num_batch, -1, 24)
#         T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
#
#         v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
#         v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
#
#         verts = v_homo[:, :, :3, 0]
#
#         joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor.T)
#         joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor.T)
#         joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor.T)
#
#         joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
#
#         # # rotate x
#         # rx_verts = torch.matmul(verts, self.Rx.T)
#         # rx_joints = torch.matmul(joints, self.Rx.T)
#
#         return verts, joints, self.faces
#
#
#
# if __name__ == '__main__':
#     smpl_path = ('D:\\paper\\human_body_reconstruction\\code\\master\\data\\basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
#     device = 'cpu'
#     smpl= SMPL(smpl_path, smpl_type='basic').to(device)
#
#     # smpl
#     pose = torch.zeros(24,3).to(device)
#     shape = torch.zeros(1,10).to(device)
#     verts, joints, faces = smpl(shape, pose)
#
#     obj = {
#         'verts': verts[0],  # 模型顶点
#         'faces': faces,  # 面片序号
#         'J': joints[0],  # 3D关节点
#     }
#
#     # 弱透视投影
#     cam = {
#         'fx': 512,
#         'fy': 512,
#         'cx': 256,
#         'cy': 180,
#         'trans_x': 0,
#         'trans_y': 0,
#         'trans_z': 2
#     }
#
#     color, depth = perspective_render_obj_debug(cam, obj, width=512, height=512,
#                                                 show_smpl_joints=True, use_viewer=True)
#
#     cv2.imshow('color', color)
#     cv2.waitKey(0)
#






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

from utils.util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose
from utils.render import perspective_render_obj_debug


class SMPL(nn.Module):
    def __init__(self, basic_model_path,
                       cocoplus_model_path,
                       weight_batch_size=128*1,
                       smpl_type='synthesis',
                       ):
        super(SMPL, self).__init__()

        if smpl_type not in ['cocoplus', 'basic', 'synthesis']:
            msg = 'unknow joint type: {}, it must be either "cocoplus", "basic" or "synthesis"'.format(smpl_type)
            sys.exit(msg)


        if smpl_type == 'basic' or smpl_type == 'synthesis':
            with open(basic_model_path, 'rb') as f:
                model_basic = pickle.load(f, encoding='iso-8859-1')
                model = model_basic

        if smpl_type == 'cocoplus' or smpl_type == 'synthesis':
            with open(cocoplus_model_path, 'rb') as f:
                model_cocoplus = pickle.load(f, encoding='iso-8859-1')
                model = model_cocoplus

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

        if smpl_type == 'basic':
            np_joint_regressor = np.array(model['J_regressor'].toarray(), dtype=np.float)
        elif smpl_type == 'cocoplus':
            np_joint_regressor = np.array(model['cocoplus_regressor'].toarray(), dtype=np.float)
        elif smpl_type == 'synthesis':
            basic_joint_regressor = np.array(model_basic['J_regressor'].toarray(), dtype=np.float)
            cocoplus_joint_regressor = np.array(model_cocoplus['cocoplus_regressor'].toarray(), dtype=np.float)
            np_joint_regressor = np.vstack((cocoplus_joint_regressor, basic_joint_regressor))
        else:
            assert 0, 'wrroy smpl type {}'.format(smpl_type)

        self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype=np.float)
        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        np_weights = np.tile(np_weights, (weight_batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None


    def save_obj(self, verts, obj_mesh_name):
        # if not self.faces:
        #     msg = 'obj not saveable!'
        #     sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


    def forward(self, beta, theta):
        device = beta.device
        self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]
        # print('smpl num_batch {}'.format(num_batch))


        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor.T)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor.T)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor.T)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)  # 减去对角线元素
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=True)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joints = torch.matmul(verts.permute(0, 2, 1), self.joint_regressor.T).permute(0, 2, 1)

        # # rotate x
        # rx_verts = torch.matmul(verts, self.Rx.T)
        # rx_joints = torch.matmul(joints, self.Rx.T)

        return verts, joints, self.faces



if __name__ == '__main__':

    basic_model_path = 'D:\\paper\\human_body_reconstruction\\code\\master\\data\\basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    cocoplus_model_path = 'D:\\paper\\human_body_reconstruction\\code\\master\\data\\neutral_smpl_with_cocoplus_reg.pkl'

    device = 'cpu'
    smpl= SMPL(basic_model_path=basic_model_path,
               cocoplus_model_path=cocoplus_model_path,
               smpl_type='synthesis').to(device)

    # smpl
    np.random.seed(9608)
    # pose = (np.random.rand(24,3) - 0.5) * 0.4
    # shape = (np.random.rand(1,10) - 0.5) * 0.06
    pose = np.zeros((24,3))
    shape = np.zeros((1,10))

    pose = torch.from_numpy(pose).to(device).type(torch.float32)
    shape = torch.from_numpy(shape).to(device).type(torch.float32)

    verts, joints, faces = smpl(shape, pose)

    smpl.save_obj(verts[0], 'test.obj')

    obj = {
        'verts': verts[0],  # 模型顶点
        'faces': faces,  # 面片序号
        'J': joints[0],  # 3D关节点
    }

    # 弱透视投影
    cam = {
        'fx': 512,
        'fy': 512,
        'cx': 256,
        'cy': 180,
        'trans_x': 0,
        'trans_y': 0,
        'trans_z': 2
    }

    color, depth = perspective_render_obj_debug(cam, obj, width=512, height=512,
                                                show_smpl_joints=True, use_viewer=True)

    cv2.imshow('smpl_cocoplus', color)

    cv2.waitKey(0)

