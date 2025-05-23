import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb, mask, edge):
        for op in self.ops:
            rgb, mask, edge = op(rgb, mask, edge)
        return rgb, mask, edge



class Normalize(object):
    def __init__(self, mean1, std1):
        self.mean1 = mean1
        self.std1  = std1

    def __call__(self, rgb, mask, edge):
        rgb = (rgb - self.mean1)/self.std1
        mask /= 255
        edge /= 255
        return rgb, mask, edge

class Minusmean(object):
    def __init__(self, mean1):
        self.mean1 = mean1

    def __call__(self,rgb, mask, edge):
        rgb = rgb - self.mean1
        mask /= 255
        edge /= 255
        return rgb, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, mask, edge):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb, mask, edge

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, mask, edge):
        H,W,_ = rgb.shape
        xmin = np.random.randint(W-self.W+1)
        ymin = np.random.randint(H-self.H+1)
        rgb = rgb[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        edge = edge[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return rgb, mask, edge

class RandomHorizontalFlip(object):
    def __call__(self, rgb, mask, edge):
        if np.random.randint(2)==1:
            rgb = rgb[:,::-1,:].copy()

            mask = mask[:,::-1,:].copy()
            edge = edge[:, ::-1, :].copy()
        return rgb, mask, edge

class ToTensor(object):
    def __call__(self, rgb, mask, edge):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)

        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)

        edge = torch.from_numpy(edge)
        edge = edge.permute(2, 0, 1)
        return rgb, mask.mean(dim=0, keepdim=True), edge.mean(dim=0, keepdim=True)