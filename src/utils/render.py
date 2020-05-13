import cv2
import numpy as np
import trimesh
import pyrender
import torch

from .util import Rx_mat

def perspective_render_obj(camera, obj, width=512, height=512, show_smpl_joints=False,
                           rotate_x_axis=True, weak_perspective=False, use_viewer=False):
    scene = pyrender.Scene()

    # add camera
    camera_pose = np.eye(4,4)
    camera_pose[:3, 3] = camera['camera_trans']
    camera=pyrender.camera.IntrinsicsCamera(
            fx=camera['fx'], fy=camera['fy'],
            cx=camera['cx'], cy=camera['cy'])
    scene.add(camera, pose=camera_pose)

    # add verts and faces
    if rotate_x_axis:
        rot_x = Rx_mat(torch.tensor([np.pi])).numpy()[0]
        obj['verts'] = np.dot(obj['verts'], rot_x.T)
        obj['J'] = np.dot(obj['J'], rot_x.T)
    if weak_perspective:
        obj['verts'][:, 2] = obj['verts'][:, 2].mean()
        obj['J'][:, 2] = obj['J'][:, 1].mean()

    vertex_colors = np.ones([obj['verts'].shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(obj['verts'], obj['faces'],
                            vertex_colors=vertex_colors)
    mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh_obj)

    # add joints
    if show_smpl_joints:
        ms = trimesh.creation.uv_sphere(radius=0.015)
        ms.visual.vertex_colors = [1.0, 0.0, 0.0]

        pts = obj['J']
        # pts = pts[22,:]

        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts

        mesh_J = pyrender.Mesh.from_trimesh(ms, poses=tfs)
        scene.add(mesh_J)

    if use_viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    # add light
    light_pose = np.eye(4,4)
    light_pose[2,3] = 2
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)
    scene.add(light, pose=light_pose)

    # render
    r = pyrender.OffscreenRenderer(viewport_width=width,viewport_height = height,point_size = 1.0)
    color, depth = r.render(scene)

    return color, depth


def weak_perspective_first_translate(verts, camera):
    '''
    对顶点做弱透视变换，只对x,y操作
    Args:
        verts:
        camera: [s,cx,cy]
    '''
    # camera = camera.view(1, 3)
    v = verts.detach().clone()

    v[..., :2] = v[..., :2] + camera[1:]
    v[..., :2] = v[..., :2] * camera[0]
    return v
