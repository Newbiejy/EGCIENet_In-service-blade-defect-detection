#coding=utf-8

import os
import cv2
import numpy as np

try:
    from lib import transform
except:
    import transform
from torch.utils.data import Dataset
# BGR
# MSRA-B

# mean_rgb = np.array([[[0.485, 0.456, 0.406]]])*255
# mean_t =np.array([[[0.485, 0.456, 0.406]]])*255
# std_rgb = np.array([[[0.229, 0.224, 0.225]]])*255
# std_t = np.array([[[0.229, 0.224, 0.225]]])*255

# VT5000

mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
# mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
# std_t = np.array([[[0.208, 0.269, 0.241]]])*255

# def getRandomSample(rgb,t):
#     n = np.random.randint(10)
#     zero = np.random.randint(2)
#     if n==1:
#         if zero:
#             rgb = torch.from_numpy(np.zeros_like(rgb))
#         else:
#             rgb = torch.from_numpy(np.random.randn(*rgb.shape))
#     elif n==2:
#         if zero:
#             t = torch.from_numpy(np.zeros_like(t))
#         else:
#             t = torch.from_numpy(np.random.randn(*t.shape))
#     return rgb,t

class Data(Dataset):
    def __init__(self, root, mode='train'):
        self.samples = []
        # lines = os.listdir(os.path.join(root, 'GT'))
        # ines = os.listdir(os.path.join(root, 'BlackWhite'))
        lines = os.listdir(os.path.join(root, 'JPEGImages'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'JPEGImages', line[:-4] + '.jpg')
            # print(line + 'pri')
            # maskpath = os.path.join(root, 'BlackWhite', line)
            # edgepath = os.path.join(root, 'Edge', line[:-4] + '.jpg')
            # self.samples.append([rgbpath, maskpath, edgepath])
            self.samples.append([rgbpath])

        if mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                               transform.Resize(352, 352),
                                               transform.RandomHorizontalFlip(),
                                               transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                               transform.Resize(352, 352),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        # rgbpath, maskpath, edgepath = self.samples[idx]
        rgbpath = self.samples[idx]
        # print(rgbpath)
        rgb = cv2.imread(rgbpath[0]).astype(np.float32)
        # t = cv2.imread(tpath).astype(np.float32)
        # mask = cv2.imread(maskpath).astype(np.float32)
        # if cv2.imread(edgepath) is None:print(idx)
        # edge = cv2.imread(edgepath).astype(np.float32)
        # H, W, C = mask.shape
        H, W, C = rgb.shape
        mask = np.zeros_like(rgb)
        edge = np.zeros_like(rgb)
        rgb, mask, edge = self.transform(rgb, mask, edge)
        # if self.mode == 'train':
        #     rgb, t =getRandomSample(rgb, t)
        # return rgb, mask, edge, (H, W), maskpath.split('/')[-1]
        return rgb, (H, W), rgbpath[0].split('/')[-1]

    def __len__(self):
        return len(self.samples)

# 合并两个dataset
class CombinedDataset(Dataset):
    def __init__(self, root1, root2, mode = 'train'):
        self.samples = []
        lines1 = os.listdir(os.path.join(root1, 'BlackWhite'))
        lines2 = os.listdir(os.path.join(root2, 'BlackWhite'))
        self.mode = mode

        for line1 in lines1:
            # rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            rgbpath = os.path.join(root1, 'JPEGImages', line1[:-4] + '.jpg')
            # print(line + 'pri')
            # maskpath = os.path.join(root, 'GT', line)
            maskpath = os.path.join(root1, 'BlackWhite', line1)
            self.samples.append([rgbpath, maskpath])

        for line2 in lines2:
            # rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            rgbpath = os.path.join(root2, 'JPEGImages', line2[:-4] + '.jpg')
            # print(line + 'pri')
            # maskpath = os.path.join(root, 'GT', line)
            maskpath = os.path.join(root2, 'BlackWhite', line2)
            self.samples.append([rgbpath, maskpath])

        if mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                               transform.Resize(352, 352),
                                               transform.RandomHorizontalFlip(),
                                               transform.ToTensor())
        elif mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                               transform.Resize(352, 352),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath, maskpath = self.samples[idx]
        # print(rgbpath)
        rgb = cv2.imread(rgbpath).astype(np.float32)
        # t = cv2.imread(tpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        rgb, mask = self.transform(rgb, mask)
        # if self.mode == 'train':
        #     rgb, t =getRandomSample(rgb, t)
        return rgb, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)