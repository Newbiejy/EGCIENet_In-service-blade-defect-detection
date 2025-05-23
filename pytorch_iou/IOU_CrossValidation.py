import os
from PIL import Image
import numpy as np
from sklearn import metrics, neighbors
import numpy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def iou(pred,gt):
    # str1 = pred_path
    # im = Image.open(str1)
    im = np.array(pred)

    # str2 = gt_path
    # im_gt = Image.open(str2)
    im_gt = np.array(gt)

    c = metrics.confusion_matrix(im.flatten(), im_gt.flatten())  # 混淆矩阵
    TP = c[1][1]  # 预测为前景，GT为前景
    TN = c[0][0]  # 预测为背景，GT为背景
    FP = c[1][0]  # 预测为前景，GT为背景
    FN = c[0][1]  # 预测为背景，GT为前景
    # print('IOU统计:\n'+str(c)+'\n\nFP为:'+str(FP)+'  FN为:'+str(FN)+'  TP为:'+str(TP)+'  TN:'+str(TN))

    iou_p = TP / (TP + FN + FP)  # 前景的IOU
    iou_n = TN / (TN + FN + FP)  # 背景的IOU
    mean_iou = (iou_n + iou_p) / 2

    # print('前景IOU:{:.3f}%\n背景IOU:{:.3f}%\nMeanIou:{:.3f}%'.format(iou_p * 100, iou_n * 100, mean_iou * 100))
    return mean_iou

def miou(pred_dir,gt_dir):
    pred = os.listdir(pred_dir)
    IOU_SUM = 0
    for i in pred:
        pred_path = os.path.join(pred_dir, i)
        gt_path = os.path.join(gt_dir, i)
        pred_img = RGBtoGRAY(pred_path)
        gt_img = Image.open(gt_path)
        IOU = iou(pred_img, gt_img)
        IOU_SUM += IOU
    return IOU_SUM/len(pred)

def miou_list(pred_dir,gt_dir):
    result = []
    pred = os.listdir(pred_dir)
    for i in pred:
        pred_path = os.path.join(pred_dir, i)
        gt_path = os.path.join(gt_dir, i.split(".")[0] + ".png")
        pred_img = RGBtoGRAY(pred_path)
        gt_img = Image.open(gt_path)
        IOU = iou(pred_img, gt_img)
        result.append([i, IOU])
        print("{} Done!".format(i))
    return result

def RGBtoGRAY(img_path):
    img = Image.open(img_path).convert("L")
    img_array = numpy.array(img)
    shape = img_array.shape
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            value = img_array[i, j]
            if value >= 128:
                img_array[i, j] = 255
            else:
                img_array[i, j] = 0
    img_result = Image.fromarray(numpy.uint8(img_array))

    return img_result

if __name__ == '__main__':
    # 这是对于不同交叉验证的方法
    pred_dir = "../output/data_587/"
    # pred_dir = "../differentView/1/"
    gt_dir = "../data_587/Test/BlackWhite/"
    mIOU = miou_list(pred_dir,gt_dir)
    print(mIOU)