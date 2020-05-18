import numpy as np
import cv2
import torch
import os
import sys
import math

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from models.network.smpl import SMPL
from .smpl_np import SMPL_np
from .render import perspective_render_obj
from .util import Clock, Rx_mat, reflect_pose, perspective_transform
from .opts import opt


class Debugger(object):
  def __init__(self, smpl_path, theme='white', down_ratio=4, device='cpu'):

    self.device = device
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)

    self.smpl = SMPL(smpl_path).to(device)

    self.names = ['p']
    self.num_joints = 19

    self.edges = [[0, 1], [1, 2], [2, 8], [7, 8], [6, 7], [8, 18], [16, 18], [14, 16],
                  [4, 5], [3, 4], [3, 9], [9, 10], [10, 11], [9, 17], [17, 15], [14, 15],
                  [8, 9], [2, 3]]

    self.ec = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
               (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
               (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
               (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
               (255, 0, 255), (255, 0, 255)]

    self.colors_hp = \
              [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0),
               (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
               (0, 0, 255), (255, 0, 0), (255, 0, 0), (255, 0, 0),
               (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 0),
               (0, 0, 255), (255, 0, 0), (0, 0, 255)]

    self.down_ratio=down_ratio

    self.camera = self.default_camera()

  def default_camera(self):
      k = np.eye(3, 3)
      k[0, 0] = 1000
      k[1, 1] = 1000
      k[0, 2] = 256
      k[1, 2] = 256

      t = np.array([[0, 0, opt.camera_pose_z]]).T

      return {
          'k': k,
          't': t
      }

  def add_blend_smpl(self, pyrender_color, img_id):
      gray = cv2.cvtColor(pyrender_color, cv2.COLOR_BGR2GRAY)
      mask =  (gray < 255).astype(np.uint8).reshape(gray.shape[0], gray.shape[1], 1)

      color_mask = pyrender_color * mask
      img_mask = self.imgs[img_id] * (1 - mask)

      self.imgs[img_id] = color_mask + img_mask

      # from PIL import Image
      # Image.fromarray(img_mask).show()



  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()


  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)


  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  
  def gen_colormap(self, img, output_res=None):
    # img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map

  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map

  def add_bbox(self, bbox, cat=0, conf=1, show_txt=True, img_id='default'):
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
      cv2.rectangle(self.imgs[img_id],
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

  def add_kp2d(self, points, img_id='default'):
    points = np.array(points, dtype=np.int32)
    for j in range(self.num_joints):
        if points[j, 2] > 0:
            cv2.circle(self.imgs[img_id],
                       (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
        if points[e, 2].sum() == 2:
            cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                     (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                     lineType=cv2.LINE_AA)


  def add_smpl_kp2d(self, pose, shape, camera, img_id='default', bbox_img_id=None):

    pose = torch.tensor(pose.reshape(24,3)).to(self.device)
    shape = torch.tensor(shape.reshape(1,10)).to(self.device)

    # smpl
    verts, joints, r, faces = self.smpl(shape, pose)

    obj = {
        'verts': verts[0],  # 模型顶点
        'faces': faces,  # 面片序号
        'J': joints[0],  # 3D关节点
    }

    # 弱透视投影
    color, depth = perspective_render_obj(camera, obj,
                   width=512, height=512, rotate_x_axis =False, show_smpl_joints=True, use_viewer=False)

    self.add_blend_smpl(color, img_id)

    ##
    rot_x = Rx_mat(torch.tensor([np.pi])).numpy()[0]
    J = np.dot(joints[0], rot_x.T)
    kp2d = perspective_transform(J, camera)
    self.add_kp2d(kp2d, bbox_img_id)



  def add_smpl(self, pose, shape, kp3d=None, camera=None, img_id='default'):

    if camera is None:
        camera = self.camera

    pose = pose.reshape(24,3).to(self.device)
    shape = shape.reshape(1,10).to(self.device)


    # smpl
    verts, joints, r, faces = self.smpl(shape, pose)

    if kp3d is not None:
        J = kp3d
    else:
        J = joints[0]

    obj = {
        'verts': verts[0],  # 模型顶点
        'faces': faces,  # 面片序号
        'J': J,  # 3D关节点
    }

    # 弱透视投影
    color, depth = perspective_render_obj(camera, obj,
                   width=512, height=512, show_smpl_joints=True, use_viewer=False)

    self.add_blend_smpl(color, img_id)

    #
    kp2d = perspective_transform(J.detach().cpu().numpy(), camera)
    self.add_kp2d(kp2d, img_id)


  def add_smpl_np(self, pose, shape, kp3d=None, camera=[1,0,0], img_id='default'):

    # smpl_np = SMPL_np("D:/paper/human_body_reconstruction/code/master/data/neutral_smpl_with_cocoplus_reg.pkl",joint_type='cocoplus')
    smpl_np = SMPL_np("D:/paper/human_body_reconstruction/code/tools/smpl_smplify/SMPL_np/model/smpl_model_male.pkl")

    pose[0:3]=1e-14
    smpl_np.set_params(beta=shape.detach().cpu().numpy().flatten(), pose=pose.detach().cpu().numpy())

    obj = smpl_np.get_obj()
    # obj['verts'] = weak_perspective(obj['verts'], camera)
    # obj['J'] = weak_perspective(obj['J'], camera)

    color_origin, depth = perspective_render_obj(obj,
                  width=512, height=512, show_smpl_joints=True)


    ## reflect pose
    # smpl_np_1 = SMPL_np("D:/paper/human_body_reconstruction/code/master/data/neutral_smpl_with_cocoplus_reg.pkl",joint_type='cocoplus')
    smpl_np_1 = SMPL_np("D:/paper/human_body_reconstruction/code/tools/smpl_smplify/SMPL_np/model/smpl_model_male.pkl")
    pose[0:3]=1e-14
    smpl_np_1.set_params(beta=shape.detach().cpu().numpy().flatten(), pose=reflect_pose(pose.detach().cpu().numpy()))

    obj_1 = smpl_np_1.get_obj()
    # obj['verts'] = weak_perspective(obj['verts'], camera)
    # obj['J'] = weak_perspective(obj['J'], camera)

    color_reflect_pose, depth = perspective_render_obj(obj_1,
                  width=512, height=512, show_smpl_joints=True)


    smpl_np.save_to_obj('D:/paper/human_body_reconstruction/code/master/src/utils/color_oringin_pose_male.obj')
    smpl_np_1.save_to_obj('D:/paper/human_body_reconstruction/code/master/src/utils/color_reflect_pose_male.obj')

    cv2.imshow('color_origin', color_origin)
    cv2.imshow('color_reflect_pose', color_reflect_pose)
    cv2.waitKey(0)

    # add image
    self.add_img(color_origin, img_id)


  def show_all_imgs(self, pause=False, time=0):
      for i, v in self.imgs.items():
         cv2.imshow('{}'.format(i), v)
      if cv2.waitKey(0 if pause else 1) == 27:
         import sys
         sys.exit(0)

      self.imgs = {}


  def save_all_imgs(self, iter_id, path):
     for i, v in self.imgs.items():
        cv2.imwrite(path + '/{}_{}.jpg'.format(iter_id, i), v)



color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
