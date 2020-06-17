import matplotlib.pyplot as plt

from collections import Counter

import numpy as np
import torch
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from .opts import opt
from .decode import decode
from .util import Rx_mat, perspective_transform
from models.network.smpl import SMPL

smpl = SMPL(opt.smpl_path).to('cpu')


def get_kp2d(pose, shape, camera):
    pose = torch.tensor(pose.reshape(24, 3)).to('cpu')
    shape = torch.tensor(shape.reshape(1, 10)).to('cpu')

    # smpl
    verts, joints, faces = smpl(shape, pose)

    ##
    rot_x = Rx_mat(torch.tensor([np.pi])).numpy()[0]
    J = np.dot(joints[0], rot_x.T)
    kp2d = perspective_transform(J, camera)

    return kp2d



def covert_eval_data(output, batch, eval_id, eval_data, data_type, score_thresh):
    # dts = []  # [img_id, confidence, kps_19x3]
    # gts = []  # [img_id, 1, kps_19x3]

    out = decode(output, thresh=score_thresh)

    dts = eval_data['dts']
    gts = eval_data['gts']

    for b in out:
        if b=={}:
            continue
        for j in range(len(b['score'])):
            conf = b['score'][j]

            if data_type == 'kps':
                kp2d = get_kp2d(b['pose'][j], b['shape'][j], b['camera'][j])
                dts.append([eval_id, conf, kp2d])
            elif data_type == 'bbox':
                bbox = b['bbox'][j].detach().cpu().numpy()
                dts.append([eval_id, conf, bbox])

    for gt in batch['gt']['gt'][0]:
        if data_type == 'kps':
            gts.append([eval_id, 1, gt['kp2d'][0].detach().cpu().numpy()])
        elif data_type == 'bbox':
            gts.append([eval_id, 1, gt['bbox'][0].detach().cpu().numpy()])


def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def _getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = Evaluator._getArea(boxA)
    area_B = Evaluator._getArea(boxB)
    if interArea is None:
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)


def iou_bbox(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou


def iou_kps(det, gt):
    mask = gt[:,2].copy()
    loss = np.abs(det[mask>0,:2] - gt[mask>0,:2])
    iou = loss.sum() / (mask.sum() + 1e-8) / 2.
    return 1.0 / iou


def get_iou(det, gt, data_type):
    iou = 0
    if data_type == 'bbox':
        iou = iou_bbox(det, gt)
    elif data_type == 'kps':
        iou = iou_kps(det, gt)
    return iou


def calculate_average_precision(rec, prec): # 每次精度变大时和变大前组成的矩形面积都将计算
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):  # 从后到前，将小于该精度位置的精度覆盖，因此形成阶梯
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):  # recall变大时的位置，这些位置precision在阶梯型增加，
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def eleven_point_interpolated_AP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]


def get_ap(eval_data, iou_thresh, data_type='kps', method='average'):
    # [img_id, confidence, kps_19x3]
    dects = eval_data['dts']
    # [img_id, 1, kps_19x3]
    gts = eval_data['gts']

    npos = len(gts)
    # sort detections by decreasing confidence
    dects = sorted(dects, key=lambda conf: conf[1], reverse=True)
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    # create dictionary with amount of gts for each image
    det = Counter([cc[0] for cc in gts])
    for key, val in det.items():
        det[key] = np.zeros(val)  # 统计每个图像gt框的个数，并分配空间
    # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
    # Loop through detections
    for d in range(len(dects)):
        # Find ground truth image
        gt = [gt for gt in gts if gt[0] == dects[d][0]]
        iouMax = sys.float_info.min
        for j in range(len(gt)):  # 得到检测框与gt框最大Iou的框
            # print('Ground truth gt => %s' % (gt[j][3],))
            iou = get_iou(dects[d][2], gt[j][2], data_type)
            if iou > iouMax:
                iouMax = iou
                jmax = j
        # Assign detection as true positive/don't care/false positive
        # print(iouMax)
        if iouMax >= iou_thresh:
            if det[dects[d][0]][jmax] == 0:
                TP[d] = 1  # count as true positive
                det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                # print("TP")
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
        # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
        else:
            FP[d] = 1  # count as false positive
            # print("FP")
    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    # Depending on the method, call the right implementation
    if method == 'average':
        [ap, mpre, mrec, ii] = calculate_average_precision(rec, prec)
    else:
        [ap, mpre, mrec, _] = eleven_point_interpolated_AP(rec, prec)
    # add class result in the dictionary to be returned
    return {
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP)
    }


def eval(eval_data, iou_thresh,
         data_type='kps',
         method='average',
         save_path=None,
         image_id='pr_curve',
         show_graphic = False,
         showInterpolatedPrecision = False):

    ret = []
    for iou_t in iou_thresh:
        result = get_ap(eval_data, iou_t, data_type, method)
        ret.append(result)

        # precision = result['precision']
        # recall = result['recall']
        # average_precision = result['AP']
        # mpre = result['interpolated precision']
        # mrec = result['interpolated recall']

    mAP = 0
    plt.close()
    plt.xlabel('recall')
    plt.ylabel('precision')

    for i, iou_t in enumerate(iou_thresh):
        if showInterpolatedPrecision:
            if method == 'average':
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == 'eleven':
                # Uncomment the line below if you want to plot the area
                # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        ap_str = "{0:.2f}%".format(100*ret[i]['AP'])
        plt.plot(ret[i]['recall'], ret[i]['precision'], label='{}-{}'.format(iou_t, ap_str))
        mAP += ret[i]['AP']

    mAP = mAP / len(ret)
    ap_str = "{0:.2f}%".format(mAP*100)
    plt.title('Precision x Recall curve of {}, mAP {}'.format(data_type, ap_str))
    plt.legend(shadow=True)
    plt.grid()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, image_id+'.png'))

    if show_graphic is True:
        plt.show()
        # plt.waitforbuttonpress()
        plt.pause(0.05)

    return mAP
