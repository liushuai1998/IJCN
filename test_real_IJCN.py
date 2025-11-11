import os.path
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
from utils import utils_image as util
from tqdm import tqdm

import torchjpeg.codec
import tempfile
import torchjpeg

from einops import rearrange


def main():
    testset_name = 'Real'  # 'Real'
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    nc = [64,128,256,512]
    nb = 4

    model_name = 'Colordouble_IJCN.pth'    # Gray_IJCN  Graydouble_IJCN   Color_IJCN   Colordouble_IJCN
    model_pool = 'model_zoo'

    testsets = 'testsets'   
    results = 'test_results'  

    result_name = 'Real' + '_' + testset_name + '_' + 'IJCN-A'   # Gray  Graydouble  Color  Colordouble
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)


    # nc = [64,128,256,512]
    # nb = 4
    show_img = False                 # default: False
    model_path = os.path.join(model_pool, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_ijcn import IJCN as net
    model = net(in_nc=3, out_nc=3, nc=nc, nb=nb, act_mode='P')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    border = 0

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnrb'] = []

    L_paths = [L_path] if os.path.isfile(L_path) else util.get_image_paths(L_path)
    # L_paths = util.get_image_paths(L_path)
    for idx, img in enumerate(tqdm(L_paths, desc=f"Processing real images", leave=False)):
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)

        img_L = cv2.cvtColor(img_H, cv2.COLOR_RGB2BGR)

        Dim, QTs, Y_coef, C_coef = torchjpeg.codec.read_coefficients(img)
        QTs = QTs.unsqueeze(0).to(device).float()
        Y_coef = Y_coef.unsqueeze(0).to(device).float()
        C_coef = C_coef.unsqueeze(0).to(device).float()

        ori_h, ori_w = Dim[0][0], Dim[0][1]

        Y_h, Y_w = Y_coef.shape[2], Y_coef.shape[3]
        C_h, C_w = C_coef.shape[2], C_coef.shape[3]
        pad_Y = 64 // 8 
        pad_C = 64 // 16
        paddingBottom_Y = (pad_Y - Y_h % pad_Y) % pad_Y
        paddingRight_Y = (pad_Y - Y_w % pad_Y) % pad_Y
        paddingBottom_C = (pad_C - C_h % pad_C) % pad_C
        paddingRight_C = (pad_C - C_w % pad_C) % pad_C

        # print('ori',QTs.shape, Y_coef.shape, C_coef.shape)

        Y_coef = dct_mirror_padding(Y_coef, paddingBottom_Y, paddingRight_Y)
        C_coef = dct_mirror_padding(C_coef, paddingBottom_C, paddingRight_C)

        # print('pad',QTs.shape, Y_coef.shape, C_coef.shape)

        img_E, _, _ = model(QTs, Y_coef, C_coef)

        img_E = img_E[:,:,:ori_h,:ori_w]

        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)

        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))


def encode_jpeg(img, quality_factor):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmpfile:
        temp_path = tmpfile.name
        
        success = cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        
        if not success:
            raise ValueError("JPEG save failed")
        
        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(temp_path)
        
    
    return dim, quantization, Y_coefficients, CbCr_coefficients


def dct_mirror_padding(dct_coef, pad_h=0, pad_w=0):
    """
    Mirror padding in the DCT domain.

    Args:
        dct_coef (Tensor): DCT coefficients with shape [bs, c, h, w, 8, 8].
        pad_h (int): number of blocks to pad vertically.
        pad_w (int): number of blocks to pad horizontally.

    Returns:
        Tensor: padded DCT coefficients with shape [bs, c, h+pad_h, w+pad_w, 8, 8].
    """  
    assert dct_coef.dim() == 6, "DCT coefficients must have 6 dimensions"
    assert dct_coef.shape[4] == 8 and dct_coef.shape[5] == 8, "DCT block size must be 8x8"

    # Create odd-index sign mask along the 64-coefficient dimension
    dct_flat = rearrange(dct_coef, 'bs c h w bh bw -> bs c (bh bw) h w')
    odd_mask = torch.ones((1, 1, 64, 1, 1), device=dct_coef.device)
    odd_mask[:, :, 1::2, :, :] = -1
    
    if pad_w > 0:
        # Horizontal mirror padding (negate odd columns)
        dct_w_mirror = torch.flip(dct_flat[:, :, :, :, -pad_w:], dims=(4,))
        # Apply odd-index sign mask along the 64-coefficient dimension
        dct_w_mirror = dct_w_mirror * odd_mask
        # Concatenate horizontally
        dct_flat = torch.cat([dct_flat, dct_w_mirror], dim=4)

    if pad_h > 0:
        # Transpose only the mirrored part (for vertical mirroring)
        dct_h_mirror = torch.flip(dct_flat[:, :, :, -pad_h:, :], dims=(3,))
        bs, c, _, h, w = dct_h_mirror.shape
        dct_h_mirror = dct_h_mirror.view(bs, c, 8, 8, h, w).transpose(2, 3).contiguous().view(bs, c, 64, h, w)
        # Apply odd-index sign mask
        dct_h_mirror = dct_h_mirror * odd_mask
        dct_h_mirror = dct_h_mirror.view(bs, c, 8, 8, h, w).transpose(2, 3).contiguous().view(bs, c, 64, h, w)
        # Concatenate vertically
        dct_flat = torch.cat([dct_flat, dct_h_mirror], dim=3)
    
    # Reshape back to [bs, c, h+pad_h, w+pad_w, 8, 8]
    dct_padded = rearrange(dct_flat, 'bs c (bh bw) h w -> bs c h w bh bw', bh=8, bw=8)

    return dct_padded

def _convert_input_type_range0(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range0(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr0(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range0(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range0(out_img, img_type)
    return out_img

if __name__ == '__main__':
    main()
