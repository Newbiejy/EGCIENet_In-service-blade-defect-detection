import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
import os
from baseline import Mnet
import numpy as np
from pytorch_iou.IOU_CrossValidation import miou

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    model_path = './model/final.pth'
    out_path = 'output/data_587/'
    if not os.path.exists(out_path): os.mkdir(out_path)
    data = Data(root='data_587/Test/', mode='test')

    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    img_num = len(loader)
    net.eval()
    time_s = time.time()


    with torch.no_grad():
        for rgb, (H, W), name in loader:
            score1, score2, score3, s1, s2, s3 = net(rgb.cuda().float())
            score1 = F.interpolate(score1, size=(H, W), mode='bilinear', align_corners=True)

            pred = np.squeeze(torch.sigmoid(score1).cpu().data.numpy())
            pred[pred > 0.5] = 1
            pred[pred < 1] = 0
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)
            print('{} Done!'.format(name))

    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))
    # 计算MIOU
    gt_dir = 'data_587/Test/BlackWhite/'
    mIOU = miou(out_path, gt_dir)
    print(mIOU)