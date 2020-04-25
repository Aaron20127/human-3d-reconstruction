import time
import cv2
import numpy as np
import copy


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
                sks = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
                       [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3],
                       [2, 4], [3, 5], [4, 6]]
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