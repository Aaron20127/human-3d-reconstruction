import numpy as np
import cv2
import random
import copy

# data color augment
_data_rng = np.random.RandomState(123)

_eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                    dtype=np.float32)
_eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def similarity_transformation_2D(p, scale, theta, trans):
    rad = np.pi * theta / 180

    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad), np.cos(rad)]])

    ts = trans.reshape((2, 1))
    ret = scale * np.dot(R, p.T) + ts

    return ret.T


def get_similarity_transform(scale, translate,
                             rotate, flip,
                             input_w, input_h):
    ## get dst
    start = np.array([[0., 0.],
                      [input_w/2., 0.],
                      [0., input_h/2.]])
    dst = similarity_transformation_2D(start, scale, rotate, translate)

    ## get src
    p0 = [input_w/2., input_h/2.]
    p1 = np.array([input_w, input_h/2.]) if not flip \
                    else np.array([0, input_h/2.])
    p2 = [input_w/2., input_h]

    src = np.stack((p0, p1, p2), 0)

    ## get mat
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return M



def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # print(src, dst)

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, M):
    """
    Argments
        pt (array,(n,2)): points
        M (array,(3,3)): matrix of affine transform
    """
    new_pt = np.hstack((pt, np.ones((pt.shape[0],1)))).T
    new_pt = np.dot(M, new_pt)
    return new_pt[:2].T


def affine_transform_kps(pt, M):
    """
    Argments
        pt (array,(n,3)): points
        M (array,(3,3)): matrix of affine transform
    """
    new_pt = np.hstack((pt[:,:2], np.ones((pt.shape[0],1)))).T
    new_pt = np.dot(M, new_pt)
    new_pt = np.hstack((new_pt.T, pt[:, 2].reshape(-1,1)))
    return new_pt


def affine_transform_bbox(bbox, t):
    p =  np.array([
        [bbox[0], bbox[1], 1.],
        [bbox[0], bbox[3], 1.],
        [bbox[2], bbox[1], 1.],
        [bbox[2], bbox[3], 1.],
    ], dtype=np.float32).T

    pt = np.dot(t, p)
    pt = pt[:2, :]

    x_min = 10000
    x_max = -10000
    y_min = 10000
    y_max = -10000
    for i in range(pt.shape[1]):
        x_min = min(x_min, pt[0,i])
        x_max = max(x_max, pt[0,i])
        y_min = min(y_min, pt[1,i])
        y_max = max(y_max, pt[1,i])

    bbox = np.array([x_min, y_min, x_max, y_max])
    return bbox


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(_data_rng, image, gs, gs_mean, 0.4)
    lighting_(_data_rng, image, 0.1, _eig_val, _eig_vec)


def addCocoAnns(anns, img, draw_skeleton=True, draw_key_points=True, draw_bbox=True):
    """
    1 . coco2017 person key points lists:
       [ 0 - 'nose',
         1 - 'left_eye',
         2 - 'right_eye',
         3 - 'left_ear',
         4 - 'right_ear',
         5 - 'left_shoulder',
         6 - 'right_shoulder',
         7 - 'left_elbow',
         8 - 'right_elbow',
         9 - 'left_wrist',
         10 - 'right_wrist',
         11 - 'left_hip',
         12 - 'right_hip',
         13 - 'left_knee',
         14 - 'right_knee',
         15 - 'left_ankle',
         16 - 'right_ankle']
    2. coco2017 person skeleton lists:
        [ [15 13],
          [13 11],
          [16 14],
          [14 12],
          [11 12],
          [ 5 11],
          [ 6 12],
          [ 5  6],
          [ 5  7],
          [ 6  8],
          [ 7  9],
          [ 8 10],
          [ 1  2],
          [ 0  1],
          [ 0  2],
          [ 1  3],
          [ 2  4],
          [ 3  5],
          [ 4  6]]
    """
    img = copy.deepcopy(img)
    if not anns:
        print('no annotations.')
        return img

    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')

    if datasetType == 'instances':
        for ann in anns:
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # sks = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
                #        [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3],
                #        [2, 4], [3, 5], [4, 6]]
                sks = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
                       [5, 7], [6, 8], [7, 9], [8, 10],  [0, 1], [0, 2], [1, 3],
                       [2, 4], [3, 5], [4, 6]] # do not use [1,2], then it looks better
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                if draw_skeleton:
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            cv2.line(img, (x[sk][0], y[sk][0]), (x[sk][1], y[sk][1]), (255, 0, 0), 2,
                                     lineType=cv2.LINE_AA)
                            # plt.plot(x[sk],y[sk], linewidth=3, color=c)
                if draw_key_points:
                    # draw_circle(img, x[v==1], y[v==1], radius = 3, color=(0,255,0))
                    # draw_circle(img, x[v==2], y[v==2], radius = 4, color=(0,0,255))
                    for i in range(len(v)):
                        if v[i] > 0:
                            pos = (int(x[i]), int(y[i]))
                            cv2.circle(img, pos, 6, (255, 0, 0), -1)

                    frontScale = 0.3
                    thickness = 1
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    for i in range(len(v)):
                        if v[i] > 0:
                            text = str(i)
                            textSize = cv2.getTextSize(text, fontFace, frontScale, thickness)[0]
                            text_w = textSize[0] / 2.0
                            text_h = textSize[1] / 2.0
                            pos_org = (int(x[i] - text_w), int(y[i] + text_h))
                            cv2.putText(img, str(i), pos_org, cv2.FONT_HERSHEY_SIMPLEX,
                                        frontScale, (255, 255, 255), thickness, cv2.LINE_AA)

            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                upper_left = (int(bbox_x), int(bbox_y))
                lower_right = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))
                cv2.rectangle(img, upper_left, lower_right, (0, 255, 255), 1)

    elif datasetType == 'captions':
        for ann in anns:
            print(ann['caption'])
    return img
