import cv2
import numpy as np
import trimesh
import pyrender


def perspective_render_obj(obj, width=512,height=512, show_smpl_joints=False, use_viewer=False):
    scene = pyrender.Scene()

    # add camera
    camera_pose = np.array([
        [1.0,  0.0,  0.0,   0],
        [0.0,  1.0,  0.0,   0],
        [0.0,  0.0,  1.0,   2],
        [0.0,  0.0,  0.0,   1.0],
    ])
    camera=pyrender.camera.IntrinsicsCamera(
            fx=width/2, fy=width/2,
            cx=width/2, cy=width/2)
    scene.add(camera, pose=camera_pose)

    # add verts and faces
    vertex_colors = np.ones([obj['verts'].shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(obj['verts'], obj['faces'],
                               vertex_colors=vertex_colors)
    mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)

    scene.add(mesh_obj)

    # add joints
    if show_smpl_joints:
        ms = trimesh.creation.uv_sphere(radius=0.15)
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
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4*camera_pose[2,3])
    scene.add(light, pose=camera_pose)

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