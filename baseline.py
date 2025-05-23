import torch
from torch import nn
import torch.nn.functional as F
import mix_transformer

import time

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class GAM(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(GAM, self).__init__()
        self.ch2 = ch_2
        self.conv_pre = convblock(ch_1, ch_2, 3, 1, 1)

    def forward(self, rgb, pre):
        cur_size = rgb.size()[2:]

        pre = self.conv_pre(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))

        fus = pre + rgb

        return fus

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # shuffle
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

'''----------CBAM----------'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class CSIM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSIM, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(in_planes=in_ch)
        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_ch))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, groups=2)
        out = out + residual

        out = self.cbam(out)
        out = self.prelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class Decoder(nn.Module):  # 解码器
    def __init__(self):
        super(Decoder, self).__init__()

        self.d31 = GAM(320, 64)
        self.d42 = GAM(512, 128)
        self.d42_31 = GAM(128, 64)  # 拟合

        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_3 = nn.Conv2d(320, 1, 1, 1, 0)
        self.score_4 = nn.Conv2d(512, 1, 1, 1, 0)

        # FEM
        self.FEM4 = CSIM(in_ch=512, out_ch=512)
        self.FEM3 = CSIM(in_ch=320, out_ch=320)
        self.FEM2 = CSIM(in_ch=128, out_ch=128)
        self.FEM1 = CSIM(in_ch=64, out_ch=64)

    def forward(self, rgb):
        d1, d2, d3, d4 = rgb[0], rgb[1], rgb[2], rgb[3]

        d4 = self.FEM4(d4)
        d3 = self.FEM3(d3)
        d2 = self.FEM2(d2)
        d1 = self.FEM1(d1)

        d13 = self.d31(d1, d3)
        d24 = self.d42(d2, d4)
        d1234 = self.d42_31(d13, d24)

        score1234 = self.score_1(d1234)
        score13 = self.score_1(d13)
        score24 = self.score_2(d24)

        return score1234, score13, score24


class Segformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(mix_transformer, backbone)()
        ## initilize encoder
        if pretrained:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

    def forward(self):
        model = Segformer('mit_b3', pretrained=True)
        return model


class Mnet(nn.Module):
    def __init__(self, backbone="mit_b3", pretrained=True):
        super(Mnet, self).__init__()

        net = Segformer(backbone, pretrained)
        self.rgb_encoder = net.encoder
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, edge=None):
        # rgb
        B = rgb.shape[0]
        Hig = rgb.shape[2]
        rgb_f = []

        if edge is not None:
            expanded_edge = edge.expand(4, 3, 352, 352)
            edge_x, edge_H, edge_W = self.rgb_encoder.patch_embed1(expanded_edge)

        # stage 1
        x, H, W = self.rgb_encoder.patch_embed1(rgb)
        for i, blk in enumerate(self.rgb_encoder.block1):
            if edge is not None:
                x = blk(x, H, W, edge_x)
            else:
                x = blk(x, H, W)
        x = self.rgb_encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 2
        x, H, W = self.rgb_encoder.patch_embed2(x)
        for i, blk in enumerate(self.rgb_encoder.block2):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 3
        x, H, W = self.rgb_encoder.patch_embed3(x)
        for i, blk in enumerate(self.rgb_encoder.block3):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 4
        x, H, W = self.rgb_encoder.patch_embed4(x)
        for i, blk in enumerate(self.rgb_encoder.block4):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        score1, score2, score3 = self.decoder(rgb_f)

        # return score1
        return score1, score2, score3, self.sigmoid(score1), self.sigmoid(score2), \
               self.sigmoid(score3)