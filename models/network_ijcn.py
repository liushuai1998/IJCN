from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import math
from models.dct import *
from models.djvit import DJViT
from einops import rearrange, repeat


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'P':
            L.append(nn.PReLU(num_parameters=out_channels))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))       
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CPC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.res(x) + x
        return out


def upsample_pixelshuffle(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


def downsample_pixelUnshuffle(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    down1 = sequential(
            nn.PixelUnshuffle(downscale_factor=int(mode[0])),
            conv(in_channels * (int(mode[0]) ** 2), out_channels, kernel_size, stride, padding, bias, mode='C', negative_slope=negative_slope))
    return down1



class QTAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CPC', negative_slope=0.2):
        super(QTAttention, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma1, beta1):     
        gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1).unsqueeze(-1)
        res = (gamma1)*self.res1(x) + beta1
        return x + res


class LOGM(nn.Module):  # Learnable Offset Guidance Module
    '''
    input: Y[bs, 1, H/8, W/8, 8, 8], C[bs, 2, H/16, W/16, 8, 8], QTs[bs, 3, 8, 8]   
    input of DJViT: Y[B, 256, H/16, W/16] CbCr[B, 128, H/16, W/16] QT[B, 384, H/16, W/16] 
    output: Y[bs, 1, H/8, W/8, 8, 8], C[bs, 2, H/16, W/16, 8, 8]
    '''
    def __init__(self, Y_channels=64, C_channels=128, kernel_size=1, stride=1, padding=0, bias=True, mode='CPC'):
        super(LOGM, self).__init__()
        self.low_freqs_Y = 56  
        self.low_freqs_C = 48  
        self.in_channels = self.low_freqs_Y *4 + self.low_freqs_C * 2
        self.learnable_offset = DJViT(dct_size=16, channel=self.in_channels)

        self.low2high_order = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
                          27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
        self.reverse_order = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
                         18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37,
                         47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]

        self.qt_embed = sequential(torch.nn.Linear(128, 512), nn.ReLU(),
                                    torch.nn.Linear(512, 512), nn.ReLU(),
                                  torch.nn.Linear(512, 512), nn.ReLU())

        self.qt_gamma = sequential(torch.nn.Linear(512, self.in_channels),nn.Sigmoid())
        self.qt_beta =  sequential(torch.nn.Linear(512, self.in_channels),nn.Tanh())


    def forward(self, Y, C, QT): 
        qt_flat = QT[:, :2, :, :].view(QT.size(0), -1).float()
        qtY = QT[:, :1, :, :].view(QT.size(0), 1, -1, 1, 1)
        qtC = QT[:, 1:, :, :].view(QT.size(0), 2, -1, 1, 1)
        qt_embed = self.qt_embed(qt_flat)
        qt_gamma = self.qt_gamma(qt_embed).unsqueeze(-1).unsqueeze(-1)
        qt_beta = self.qt_beta(qt_embed).unsqueeze(-1).unsqueeze(-1)

        Y = rearrange(Y, 'b c (h b1) (w b2) mh mw -> b (c b1 b2) (mh mw) h w', b1=2, b2=2)  # [bs, 4, 64, H/16, W/16]
        C = rearrange(C, 'b c h w mh mw -> b c (mh mw) h w')  # [bs, 2, 64, H/16, W/16]
        Yl = (Y*qtY)[:, :, self.low2high_order, :, :][:, :, :self.low_freqs_Y, :, :]  # [bs, 4, 56, H/16, W/16]
        Cl = (C*qtC)[:, :, self.low2high_order, :, :][:, :, :self.low_freqs_C, :, :]  # [bs, 2, 48, H/16, W/16]
        Yl = rearrange(Yl, 'b c n h w -> b (c n) h w')  # [bs, 224, H/16, W/16]
        Cl = rearrange(Cl, 'b c n h w -> b (c n) h w')  # [bs, 96, H/16, W/16]

        Dl = torch.cat([Yl, Cl], dim=1)   # [bs, 320, H/16, W/16]

        Dl = qt_gamma * self.learnable_offset(Dl) + qt_beta

        Yl_offset, Cl_offset = torch.split(Dl, [self.low_freqs_Y*4, self.low_freqs_C*2], dim=1)
        Yl_offset = rearrange(Yl_offset, 'b (c n) h w -> b c n h w', c=4)
        Cl_offset = rearrange(Cl_offset, 'b (c n) h w -> b c n h w', c=2)
        Y_offset = torch.zeros_like(Y)
        C_offset = torch.zeros_like(C)
        Y_offset[:, :, :self.low_freqs_Y, :, :] = Yl_offset
        C_offset[:, :, :self.low_freqs_C, :, :] = Cl_offset
        Y_offset = Y_offset[:, :, self.reverse_order, :, :]
        C_offset = C_offset[:, :, self.reverse_order, :, :]

        Y_pred = 0.5 * torch.tanh(Y_offset) + Y
        C_pred = 0.5 * torch.tanh(C_offset) + C

        Y_pred = rearrange(Y_pred, 'b (c b1 b2) (mh mw) h w -> b c (h b1) (w b2) mh mw', c=1, b1=2, b2=2, mh=8, mw=8)
        C_pred = rearrange(C_pred, 'b c (mh mw) h w -> b c h w mh mw', mh=8, mw=8)

        Y = Y_pred * QT[:, :1, :, :].unsqueeze(2).unsqueeze(3)
        C = C_pred * QT[:, 1:, :, :].unsqueeze(2).unsqueeze(3)

        return Y, C, Y_pred, C_pred


class IJCN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='P'):
        super(IJCN, self).__init__()
        self.nb = nb
        self.nc = nc
        downsample_block = downsample_pixelUnshuffle
        upsample_block = upsample_pixelshuffle
        
        self.DCT_offset = LOGM(64, 128, bias=True)
        self.Pixel_up =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.m_down1_QT = QTAttention(nc[0], nc[0], 1,1,0, bias=True, mode='C' + act_mode + 'C')
        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(4)],
        downsample_pixelUnshuffle(nc[0], nc[2], bias=True, mode='2'))

        self.m_down2_QT = QTAttention(nc[2], nc[2], 1,1,0, bias=True, mode='C' + act_mode + 'C')
        self.m_down2 = sequential(*[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)],
        downsample_pixelUnshuffle(nc[2], nc[2], 1,1,0, bias=True, mode='2'))

        self.m_down3_QT = QTAttention(nc[2], nc[2], 1,1,0, bias=True, mode='C' + act_mode + 'C')
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])
        
        self.m_down4 = sequential(*[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])

        self.m_up4 = sequential(*[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])

        self.m_up3 = sequential(*[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])
        self.m_up3_QT = QTAttention(nc[2], nc[2], 1,1,0, bias=True, mode='C' + act_mode + 'C')

        self.m_up2 = sequential(upsample_pixelshuffle(nc[2], nc[2], 1,1,0, bias=True, mode='2'),
        *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])
        self.m_up2_QT = QTAttention(nc[2], nc[2], 1,1,0, bias=True, mode='C' + act_mode + 'C')

        self.m_up1 = sequential(upsample_pixelshuffle(nc[2], nc[0], bias=True, mode='2'),
        *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(4)])
        self.m_up1_QT = QTAttention(nc[0], nc[0], 1,1,0, bias=True, mode='C' + act_mode + 'C')

        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')


        self.qf_embed1i = sequential(torch.nn.Linear(128, 512), nn.ReLU(),
                                    torch.nn.Linear(512, 512), nn.ReLU(),
                                  torch.nn.Linear(512, 512), nn.ReLU())

        self.qf_gamma1_3i = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.qf_beta1_3i =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.qf_gamma1_2i = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.qf_beta1_2i =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.qf_gamma1_1i = sequential(torch.nn.Linear(512, nc[0]),nn.Sigmoid())
        self.qf_beta1_1i =  sequential(torch.nn.Linear(512, nc[0]),nn.Tanh())

        self.qf_embed1o = sequential(torch.nn.Linear(128, 512), nn.ReLU(),
                                    torch.nn.Linear(512, 512), nn.ReLU(),
                                  torch.nn.Linear(512, 512), nn.ReLU())

        self.qf_gamma1_3o = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.qf_beta1_3o =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.qf_gamma1_2o = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.qf_beta1_2o =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.qf_gamma1_1o = sequential(torch.nn.Linear(512, nc[0]),nn.Sigmoid())
        self.qf_beta1_1o =  sequential(torch.nn.Linear(512, nc[0]),nn.Tanh())

        
    def forward(self, qts, y_coef, c_coef):
        qt_flat = qts[:, :2, :, :].view(qts.size(0), -1).float()

        y_coef = y_coef.float()
        c_coef = c_coef.float()

        y_coef, c_coef, Y_pred, C_pred = self.DCT_offset(y_coef, c_coef, qts)
        y_img, c_img = batch_idct(y_coef, c_coef, device=qts.device) 
        c_img_up = self.Pixel_up(c_img)
        xinput = torch.cat([y_img, c_img_up], dim=1)
        x_input = batch_to_images(xinput, device=xinput.device)

        qf_embed1i = self.qf_embed1i(qt_flat)
        qf_embed1o = self.qf_embed1o(qt_flat)

        qf_gamma1_1i = self.qf_gamma1_1i(qf_embed1i)
        qf_beta1_1i = self.qf_beta1_1i(qf_embed1i)
        qf_gamma1_2i = self.qf_gamma1_2i(qf_embed1i)
        qf_beta1_2i = self.qf_beta1_2i(qf_embed1i)
        qf_gamma1_3i = self.qf_gamma1_3i(qf_embed1i)
        qf_beta1_3i = self.qf_beta1_3i(qf_embed1i)  

        qf_gamma1_1o = self.qf_gamma1_1o(qf_embed1o)
        qf_beta1_1o = self.qf_beta1_1o(qf_embed1o)
        qf_gamma1_2o = self.qf_gamma1_2o(qf_embed1o)
        qf_beta1_2o = self.qf_beta1_2o(qf_embed1o)
        qf_gamma1_3o = self.qf_gamma1_3o(qf_embed1o)
        qf_beta1_3o = self.qf_beta1_3o(qf_embed1o) 
        
        x1 = self.m_head(x_input)
        x1_QT = self.m_down1_QT(x1, qf_gamma1_1i, qf_beta1_1i)
        x2 = self.m_down1(x1_QT)
        x2_QT = self.m_down2_QT(x2, qf_gamma1_2i, qf_beta1_2i)
        x3 = self.m_down2(x2_QT)
        x3_QT = self.m_down3_QT(x3, qf_gamma1_3i, qf_beta1_3i)
        x4 = self.m_down3(x3_QT)
        
        x = self.m_down4(x4)

        x = self.m_up4(x)
        x = x + x4
        
        x = self.m_up3(x)
        x = self.m_up3_QT(x, qf_gamma1_3o, qf_beta1_3o)
        x = x + x3
        x = self.m_up2(x)
        x = self.m_up2_QT(x, qf_gamma1_2o, qf_beta1_2o)
        x = x + x2
        x = self.m_up1(x)
        x = self.m_up1_QT(x, qf_gamma1_1o, qf_beta1_1o)

        x = self.m_tail(x) + x_input

        return x, Y_pred, C_pred
