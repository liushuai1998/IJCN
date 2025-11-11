import torch
import math
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

def deblockify(Y_img, C_img):

    Y = rearrange(Y_img, 'b c h w bh bw -> b c (h bh) (w bw)')
    C = rearrange(C_img, 'b c h w bh bw -> b c (h bh) (w bw)')

    return Y, C


def batch_idct(Y_dct, C_dct, device=None):  # idct Y+CbCr [0,255]

    Y_img = block_idct(Y_dct, device=device)
    C_img = block_idct(C_dct, device=device)

    Y = rearrange(Y_img, 'b c h w bh bw -> b c (h bh) (w bw)')
    C = rearrange(C_img, 'b c h w bh bw -> b c (h bh) (w bw)')
    Y = (Y + 128)/255.0
    C = (C + 128)/255.0

    return Y, C    # Y[bs, 1, H, W] C[bs, 2, H/2, W/2]


def batch_idct_y(Y_dct, device=None):  # idct Y+CbCr [0,255]

    Y_img = block_idct(Y_dct, device=device)

    Y = rearrange(Y_img, 'b c h w bh bw -> b c (h bh) (w bw)')
    Y = (Y + 128)/255.0

    return Y    # Y[bs, 1, H, W] C[bs, 2, H/2, W/2]


def batch_to_images(img_est, device=None):
    img_est = img_est * 255
    if img_est.shape[1] == 3:
        img_est = to_rgb(img_est, device)

    img_est = img_est.clamp(0, 255)
    img_est = img_est / 255

    return img_est


def normalize(N):
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return (n @ n.t())


def harmonics(N):
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(im, device=None):
    N = im.shape[-1]

    n = normalize(N)
    h = harmonics(N)

    if device is not None:
        n = n.to(device)
        h = h.to(device)

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ im @ h)

    return coeff


def block_idct(coeff, device=None):
    N = coeff.shape[-1]

    n = normalize(N)
    h = harmonics(N)

    if device is not None:
        n = n.to(device)
        h = h.to(device)

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def to_ycbcr(x, device=None):
    ycbcr_from_rgb = torch.Tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ]).view(3, 3).transpose(0, 1).to(device)

    b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

    x = torch.einsum('cv,bcxy->bvxy', [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def to_rgb(x, device=None):
    rgb_from_ycbcr = torch.Tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ]).view(3, 3).transpose(0, 1).to(device)

    b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

    x -= b
    x = torch.einsum('cv,bcxy->bvxy', [rgb_from_ycbcr, x])

    return x.contiguous()
