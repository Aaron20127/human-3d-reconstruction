from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from random import randint
import pickle
from scipy.io  import loadmat
import scipy.spatial.distance
import json
import copy
from mpl_toolkits.mplot3d import axes3d, Axes3D

abspath = os.path.abspath(os.path.dirname(__file__))


class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat(abspath + '/../../data/UV_Processed.mat')
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24,
                                    23]
        UV_symmetry_filename = abspath + '/../../data/UV_symmetry_transforms.mat'
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        ###
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                ###
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        #
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        #
        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return (1 - (r + t), r, t)

    def IUV2FBC(self, I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],
                         np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],
                         np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],
                         np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()
        #

        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            if (self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return (FaceIndicesNow[0][i], bc1, bc2, bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if ((minD1 < minD2) & (minD1 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.)
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.)
        else:
            return (FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.)

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        ## 得到3角面片的3个顶点索引
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        ## 3个顶点值求和，得到最后的顶点值,bc1+bc2+bc3=1
        p = Vertices[Vert_indices[0], :] * bc1 + \
            Vertices[Vert_indices[1], :] * bc2 + \
            Vertices[Vert_indices[2], :] * bc3
        ##
        return (p)

    def get_smpl_verts_indices(self, FaceIndex):
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        return Vert_indices



def demo_visualize_densepose_label():
    coco_folder = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2014'
    # dp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json')
    dp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')
    # Get img id's for the minival dataset.
    im_ids = dp_coco.getImgIds()
    # Select a random image id.
    Selected_im = im_ids[randint(0, len(im_ids))] # Choose im no 57 to replicate
    # Load the image
    im = dp_coco.loadImgs(Selected_im)[0]
    # Load Anns for the selected image.
    ann_ids = dp_coco.getAnnIds( imgIds=im['id'] )
    anns = dp_coco.loadAnns(ann_ids)
    # Now read and b
    im_name = os.path.join( coco_folder + '/train2014', im['file_name'] )
    I=cv2.imread(im_name)
    plt.figure(1)
    plt.imshow(I[:,:,::-1]); plt.axis('off');



    ### 1. part get segmentation.
    def GetDensePoseMask(Polys):
        MaskGen = np.zeros([256,256])
        for i in range(1,15):
            if(Polys[i-1]):
                current_mask = mask_util.decode(Polys[i-1])
                MaskGen[current_mask>0] = i
        return MaskGen

    I_vis=(I.copy()/2).astype(np.uint8) # Dim the image.

    for ann in anns:
        bbr =  np.array(ann['bbox']).astype(int) # the box.
        if( 'dp_masks' in ann.keys()): # If we have densepose annotation for this ann,
            Mask = GetDensePoseMask(ann['dp_masks'])
            ################
            x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
            x2 = min( [ x2,I.shape[1] ] );  y2 = min( [ y2,I.shape[0] ] )
            ################
            MaskIm = cv2.resize( Mask, (int(x2-x1),int(y2-y1)) ,interpolation=cv2.INTER_NEAREST)
            MaskBool = np.tile((MaskIm==0)[:,:,np.newaxis],[1,1,3])
            #  Replace the visualized mask image with I_vis.
            Mask_vis = cv2.applyColorMap( (MaskIm*15).astype(np.uint8) , cv2.COLORMAP_PARULA)[:,:,:]
            Mask_vis[MaskBool]=I_vis[y1:y2,x1:x2,:][MaskBool]
            I_vis[y1:y2,x1:x2,:] = I_vis[y1:y2,x1:x2,:]*0.3 + Mask_vis*0.7

    plt.figure(2)
    plt.imshow(I_vis[:,:,::-1]); plt.axis('off');

    ### 2. show iuv.
    # Show images for each subplot.
    fig = plt.figure(3,figsize=[15, 5])
    plt.subplot(1, 3, 1)
    plt.imshow(I[:, :, ::-1]);
    plt.axis('off');
    plt.title('Patch Indices')
    plt.subplot(1, 3, 2)
    plt.imshow(I[:, :, ::-1]);
    plt.axis('off');
    plt.title('U coordinates')
    plt.subplot(1, 3, 3)
    plt.imshow(I[:, :, ::-1]);
    plt.axis('off');
    plt.title('V coordinates')

    ## For each ann, scatter plot the collected points.
    for ann in anns:
        bbr = np.round(ann['bbox'])
        if ('dp_masks' in ann.keys()):
            ## 图片中标注点的位置，是bbox的相对坐标
            Point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
            Point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.

            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min([x2, I.shape[1]])
            y2 = min([y2, I.shape[0]])
            ## 图片中标注点的位置，得到图像中的坐标
            Point_x = Point_x + x1
            Point_y = Point_y + y1


            Point_I = np.array(ann['dp_I'])
            Point_U = np.array(ann['dp_U'])
            Point_V = np.array(ann['dp_V'])

            plt.subplot(1, 3, 1)
            plt.scatter(Point_x, Point_y, 22, Point_I)
            plt.subplot(1, 3, 2)
            plt.scatter(Point_x, Point_y, 22, Point_U)
            plt.subplot(1, 3, 3)
            plt.scatter(Point_x, Point_y, 22, Point_V)

    plt.show()


def demo_matching_with_smpl():
    def smpl_view_set_axis_full_body(ax, azimuth=0):
        ## Manually set axis
        ax.view_init(0, azimuth)
        max_range = 0.55
        ax.set_xlim(- max_range, max_range)
        ax.set_ylim(- max_range, max_range)
        ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
        ax.axis('off')


    def smpl_view_set_axis_face(ax, azimuth=0):
        ## Manually set axis
        ax.view_init(0, azimuth)
        max_range = 0.1
        ax.set_xlim(- max_range, max_range)
        ax.set_ylim(- max_range, max_range)
        ax.set_zlim(0.45 - max_range, 0.45 + max_range)
        ax.axis('off')

    # Now read the smpl model.
    with open(abspath + '/../../data/neutral_smpl_with_cocoplus_reg.pkl', 'rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
        Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
        X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]
    ## Now let's rotate around the model and zoom into the face.

    fig = plt.figure(1, figsize=[16, 4])

    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax)

    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax, 45)

    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax, 90)

    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(Z, X, Y, s=0.2, c='k')
    smpl_view_set_axis_face(ax, -40)

    # plt.show()

    ## 2. show matching points
    DP = DensePoseMethods()
    pkl_file = open('D:\\paper\\human_body_reconstruction\\code\\experiment\\human_3d_reconstruction\\DensePose\\DensePose-master\\DensePoseData\\demo_data/demo_dp_single_ann.pkl', 'rb')
    Demo = pickle.load(pkl_file, encoding='iso-8859-1')

    collected_x = np.zeros(Demo['x'].shape)
    collected_y = np.zeros(Demo['x'].shape)
    collected_z = np.zeros(Demo['x'].shape)

    for i, (ii, uu, vv) in enumerate(zip(Demo['I'], Demo['U'], Demo['V'])):
        # Convert IUV to FBC (faceIndex and barycentric coordinates.)
        FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
        # Use FBC to get 3D coordinates on the surface.
        p = DP.FBC2PointOnSurface(FaceIndex, bc1, bc2, bc3, Vertices)
        #
        collected_x[i] = p[0]
        collected_y[i] = p[1]
        collected_z[i] = p[2]

    fig = plt.figure(2, figsize=[15,5])

    # Visualize the image and collected points.
    ax = fig.add_subplot(131)
    ax.imshow(Demo['ICrop'])
    ax.scatter(Demo['x'],Demo['y'],11, np.arange(len(Demo['y']))  )
    plt.title('Points on the image')
    ax.axis('off'),

    ## Visualize the full body smpl male template model and collected points
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(Z,X,Y,s=0.02,c='k')
    ax.scatter(collected_z,  collected_x,collected_y,s=25,  c=  np.arange(len(Demo['y']))    )
    smpl_view_set_axis_full_body(ax)
    plt.title('Points on the SMPL model')

    ## Now zoom into the face.
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(Z,X,Y,s=0.2,c='k')
    ax.scatter(collected_z,  collected_x,collected_y,s=55,c=np.arange(len(Demo['y'])))
    smpl_view_set_axis_face(ax)
    plt.title('Points on the SMPL model')
    #
    plt.show()


def test_dense_points(img, dense_points):
    def smpl_view_set_axis_full_body(ax, azimuth=0):
        ## Manually set axis
        ax.view_init(0, azimuth)
        max_range = 0.55
        ax.set_xlim(- max_range, max_range)
        ax.set_ylim(- max_range, max_range)
        ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
        ax.axis('off')

    def smpl_view_set_axis_face(ax, azimuth=0):
        ## Manually set axis
        ax.view_init(0, azimuth)
        max_range = 0.1
        ax.set_xlim(- max_range, max_range)
        ax.set_ylim(- max_range, max_range)
        ax.set_zlim(0.45 - max_range, 0.45 + max_range)
        ax.axis('off')

        # Now read the smpl model.

    with open(abspath + '/../../data/neutral_smpl_with_cocoplus_reg.pkl', 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
        X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]
        ## Now let's rotate around the model and zoom into the face.

    fig = plt.figure(1, figsize=[16, 4])

    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax)

    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax, 45)

    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax, 90)

    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(Z, X, Y, s=0.2, c='k')
    smpl_view_set_axis_face(ax, -40)

    # plt.show()

    ## 2. show matching points
    # DP = DensePoseMethods()
    # pkl_file = open(
    #     'D:\\paper\\human_body_reconstruction\\code\\experiment\\human_3d_reconstruction\\DensePose\\DensePose-master\\DensePoseData\\demo_data/demo_dp_single_ann.pkl',
    #     'rb')
    # Demo = pickle.load(pkl_file, encoding='iso-8859-1')

    collected_x = np.zeros(dense_points['pts_2d'][0].shape)
    collected_y = np.zeros(dense_points['pts_2d'][0].shape)
    collected_z = np.zeros(dense_points['pts_2d'][0].shape)

    for i in range(len(dense_points['pts_2d'][0])):
        # Convert IUV to FBC (faceIndex and barycentric coordinates.)
        # FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
        # Use FBC to get 3D coordinates on the surface.
        # p = DP.FBC2PointOnSurface(FaceIndex, bc1, bc2, bc3, Vertices)
        #

        ## 3个顶点值求和，得到最后的顶点值,bc1+bc2+bc3=1
        p = Vertices[dense_points['v_ind'][i][0], :] * dense_points['v_rat'][i][0] + \
            Vertices[dense_points['v_ind'][i][1], :] * dense_points['v_rat'][i][1] + \
            Vertices[dense_points['v_ind'][i][2], :] * dense_points['v_rat'][i][2]

        collected_x[i] = p[0]
        collected_y[i] = p[1]
        collected_z[i] = p[2]

    fig = plt.figure(2, figsize=[15, 5])

    # Visualize the image and collected points.
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.scatter(dense_points['pts_2d'][0], dense_points['pts_2d'][1], 11, np.arange(len(dense_points['pts_2d'][0])))
    plt.title('Points on the image')
    ax.axis('off'),

    ## Visualize the full body smpl male template model and collected points
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    ax.scatter(collected_z, collected_x, collected_y, s=25, c=np.arange(len(dense_points['pts_2d'][0])))
    smpl_view_set_axis_full_body(ax)
    plt.title('Points on the SMPL model')

    ## Now zoom into the face.
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(Z, X, Y, s=0.2, c='k')
    ax.scatter(collected_z, collected_x, collected_y, s=55, c=np.arange(len(dense_points['pts_2d'][0])))
    smpl_view_set_axis_face(ax)
    plt.title('Points on the SMPL model')
    #
    plt.show()

def pack_dense_points(ann):
    bbr = np.round(ann['bbox'])
    if ('dp_masks' in ann.keys()):
        gt = {}

        ## 图片中标注点的位置，是bbox的相对坐标
        Point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
        Point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.

        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        ## gt image points
        Point_x = Point_x + x1
        Point_y = Point_y + y1

        gt['pts_2d'] = np.stack((Point_x, Point_y), 1).flatten().tolist() # nx3

        ## gt smpl points index
        DP = DensePoseMethods()

        v_ind = np.zeros([len(ann['dp_I']), 3], dtype=np.int64)
        v_rat = np.zeros([len(ann['dp_I']), 3])

        for i, (ii, uu, vv) in enumerate(zip(ann['dp_I'], ann['dp_U'], ann['dp_V'])):
            # Convert IUV to FBC (faceIndex and barycentric coordinates.)
            FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
            verts_index = DP.get_smpl_verts_indices(FaceIndex)

            v_ind[i] = verts_index
            v_rat[i] = [bc1, bc2, bc3]

        gt['v_ind'] = v_ind.flatten().tolist() # nx3
        gt['v_rat'] = v_rat.flatten().tolist()  # nx3

        return gt


def data_prepare():
    coco_folder = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2014'
    coco_2014_dp_file = coco_folder + '/annotations/densepose_coco_2014_train.json'
    coco_2014_file = coco_folder + '/annotations/person_keypoints_train2014.json'
    dst_file = coco_folder + '/annotations/densepose_person_keypoints_train2014.json'


    coco_2014 = COCO(coco_2014_file)
    coco_2014_dp = COCO(coco_2014_dp_file)

    # compute max num pts, max==184
    # im_ids = coco_2014_dp.getImgIds()
    # max_num_pts = 0
    # for i, im_id in enumerate(im_ids):
    #     print('{} / {}'.format(i, len(im_ids)))
    #     idxs_dp = coco_2014_dp.getAnnIds(imgIds=[im_id], catIds=1, iscrowd=0)
    #     for idx_dp in idxs_dp:
    #         ann_dp = coco_2014_dp.loadAnns(ids=idx_dp)
    #         if 'dp_masks' in ann_dp[0].keys():
    #             if max_num_pts < len(ann_dp[0]['dp_I']):
    #                 max_num_pts = len(ann_dp[0]['dp_I'])
    # print('max_num_pts: {}'.format(max_num_pts))

    ##
    im_ids = coco_2014_dp.getImgIds()
    for i, im_id in enumerate(im_ids):
        print('{} / {}'.format(i, len(im_ids)))
        idxs_dp = coco_2014_dp.getAnnIds(imgIds=[im_id], catIds=1, iscrowd=0)

        idxs = coco_2014.getAnnIds(imgIds=[im_id], catIds=1, iscrowd=0)
        # anns = coco_2014.loadAnns(ids=idxs)

        for idx_dp in idxs_dp:
            for idx in idxs:
                if idx_dp == idx:
                    ann_dp = coco_2014_dp.loadAnns(ids=idx_dp)

                    if 'dp_masks' in ann_dp[0].keys():
                        ann = coco_2014.loadAnns(ids=idx)

                        ann[0]['dense_points'] = pack_dense_points(ann_dp[0])

                        # im = coco_2014_dp.loadImgs(im_id)[0]
                        # im_name = os.path.join(coco_folder + '/train2014', im['file_name'])
                        # I = cv2.imread(im_name)
                        #
                        # test_dense_points(I, ann[0]['dense_points'])
                        break
                    # dense_points = get_dens_points(ann_dp)

        if (i+1) % 10000 == 0:
            dst_data = coco_2014.get_dataset()
            dst_f = open(dst_file, 'w')
            json.dump(dst_data, dst_f)
            dst_f.close()
            print('save {}'.format(i))


    ### save
    dst_data = coco_2014.get_dataset()
    dst_f = open(dst_file, 'w')
    json.dump(dst_data, dst_f)
    dst_f.close()


if __name__ == '__main__':
    # demo_visualize_densepose_label()
    # demo_matching_with_smpl()
    data_prepare()