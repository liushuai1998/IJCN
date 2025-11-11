import os 
import os.path
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
from utils import utils_image as util
from einops import rearrange
from tqdm import tqdm

import torchjpeg.codec
import tempfile
import torchjpeg


def main():
    qf1_list = [30, 50]
    qf2_list = [30, 50]
    shift = [[0, 0], [4, 4]]
    testsets_name = ['LIVE1_color', 'urban100'] 
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    nc = [64,128,256,512]
    nb = 4
    model_name = 'Colordouble_IJCN.pth'   # Color_IJCN   Colordouble_IJCN
    model_pool = 'model_zoo'   # fixed
    testsets = 'testsets'   
    results = 'test_results'     

    model_path = os.path.join(model_pool, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_ijcn import IJCN as net
    model = net(in_nc=3, out_nc=3, nc=nc, nb=nb, act_mode='P')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        raise FileNotFoundError(f'Model file not found: {model_path}')

    for testset_name in testsets_name:
        result_name = 'Colordouble' + '_' + testset_name + '_' + 'IJCN-A'   # Color  Colordouble
        H_path = os.path.join(testsets, testset_name)
        E_path = os.path.join(results, result_name)  # E_path, for Estimated images
        util.mkdir(E_path)
        f = open(os.path.join(E_path, f'{result_name}.txt'), 'a')
        print('model_path', model_path, sep=',', end='\n', file=f)

        ave_results = OrderedDict()
        ave_results['qf_shift'] = [] 
        ave_results['ave_psnr'] = []
        ave_results['ave_ssim'] = []
        ave_results['ave_psnrb'] = []

        for qf1 in qf1_list:
            for qf2 in qf2_list:
                E_img_subfolder = os.path.join(E_path, str(qf1)+'_'+str(qf2))
                util.mkdir(E_img_subfolder)
                
                for (shift_h, shift_w) in shift:
                    E_img_subsubfolder = os.path.join(E_img_subfolder, str(shift_h)+'_'+str(shift_w))
                    util.mkdir(E_img_subsubfolder)

                    border = 0
                    test_results = OrderedDict()
                    test_results['psnr'] = []
                    test_results['ssim'] = []
                    test_results['psnrb'] = []

                    H_paths = util.get_image_paths(H_path)
                    for idx, img in enumerate(tqdm(H_paths, desc=f"Processing {testset_name} QF{qf1}_{qf2} shift{shift_h}_{shift_w}", leave=False)):
                        img_name, ext = os.path.splitext(os.path.basename(img))
                        img_H = util.imread_uint(img, n_channels=n_channels)
                        img_L = cv2.cvtColor(img_H, cv2.COLOR_RGB2BGR)
                        
                        _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), qf1])
                        img_L = cv2.imdecode(encimg, 1)

                        img_L = img_L[shift_h:,shift_w:]
                        img_H = img_H[shift_h:,shift_w:].squeeze()

                        ori_h, ori_w = img_L.shape[:2]
                        pad = 64
                        paddingBottom = (pad - ori_h % pad) % pad
                        paddingRight = (pad - ori_w % pad) % pad
                        img_L = cv2.copyMakeBorder(img_L, 0, paddingBottom, 0, paddingRight, cv2.BORDER_REFLECT)
                                    
                        h_pad1, w_pad1 = img_L.shape[:2]
                        if testset_name == 'ICB_color':
                            window_size = 512
                            crop_size = 384

                            pad2 = (window_size-crop_size) // 2
                            paddingBottom2 = (crop_size - h_pad1 % crop_size) % crop_size
                            paddingRight2 = (crop_size - w_pad1 % crop_size) % crop_size
                            img_L = cv2.copyMakeBorder(img_L, pad2, paddingBottom2 + pad2, pad2, paddingRight2 + pad2, cv2.BORDER_REFLECT)

                            Dim, QTs, Y_coef, C_coef = encode_jpeg(img_L, qf2)
                            _, dct_h, dct_w, _, _ = Y_coef.shape
                            QTs = QTs.unsqueeze(0).to(device).float()
                            Y_coef = Y_coef.unsqueeze(0).to(device).float()
                            Y_coef = rearrange(Y_coef, 'bs c h w bh bw -> bs (c bh bw) h w')
                            C_coef = C_coef.unsqueeze(0).to(device).float()
                            C_coef = rearrange(C_coef, 'bs c h w bh bw -> bs (c bh bw) h w')

                            Y_coef_blocks = extract_overlapping_blocks(Y_coef, window_size//8, crop_size//8, channels=64)
                            C_coef_blocks = extract_overlapping_blocks(C_coef, window_size//16, crop_size//16, channels=128)
                            Y_coef_blocks = rearrange(Y_coef_blocks, 'bs (c bh bw) h w -> bs c h w bh bw', c=1, bh=8, bw=8)
                            C_coef_blocks = rearrange(C_coef_blocks, 'bs (c bh bw) h w -> bs c h w bh bw', c=2, bh=8, bw=8)

                            img_E = []
                            
                            with torch.no_grad():
                                for i in range(len(Y_coef_blocks)):
                                    block_y = Y_coef_blocks[i:i+1]
                                    block_cbcr = C_coef_blocks[i:i+1]

                                    img_Ei, _, _ = model(QTs, block_y, block_cbcr)

                                    start = (window_size - crop_size) // 2
                                    end = start + crop_size
                                    img_E.append(img_Ei[:, :, start:end, start:end])
                                    
                                    torch.cuda.empty_cache()
                            img_E = reconstruct_image(img_E, (h_pad1+paddingBottom2)//crop_size, (w_pad1+paddingRight2)//crop_size)

                        else:
                            Dim, QTs, Y_coef, C_coef = encode_jpeg(img_L, qf2)
                            QTs = QTs.unsqueeze(0).to(device).float()
                            Y_coef = Y_coef.unsqueeze(0).to(device).float()
                            C_coef = C_coef.unsqueeze(0).to(device).float()

                            img_E, _, _ = model(QTs, Y_coef, C_coef)


                        img_E = img_E[:,:,:ori_h,:ori_w]

                        img_E = util.tensor2single(img_E)
                        img_E = util.single2uint(img_E)

                        # --------------------------------
                        # PSNR and SSIM, PSNRB
                        # --------------------------------
                        psnr = util.calculate_psnr(img_E, img_H, border=border)
                        ssim = util.calculate_ssim(img_E, img_H, border=border)
                        psnrb = util.calculate_psnrb(img_H, img_E, border=border)
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)
                        test_results['psnrb'].append(psnrb)

                        # util.imsave(img_E, os.path.join(E_img_subsubfolder, img_name+'.png'))
                        print(str(qf1)+'_'+str(qf2)+'_'+str(shift_h)+'_'+str(shift_w), img_name, f"{psnr:.2f}", f"{ssim:.3f}", f"{psnrb:.2f}", sep=',', end='\n', file=f)
            
                    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
                    ave_results['ave_psnr'].append(ave_psnr)
                    ave_results['ave_ssim'].append(ave_ssim)
                    ave_results['ave_psnrb'].append(ave_psnrb)
                    ave_results['qf_shift'].append(str(qf1)+'_'+str(qf2)+'_'+str(shift_h)+'_'+str(shift_w))

        for i in range(len(ave_results['ave_psnr'])):
            print(ave_results['qf_shift'][i], f"{ave_results['ave_psnr'][i]:.2f}", f"{ave_results['ave_ssim'][i]:.3f}", f"{ave_results['ave_psnrb'][i]:.2f}", sep=',', end='\n', file=f)
        f.close()


def encode_jpeg(img, quality_factor):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmpfile:
        temp_path = tmpfile.name 
        
        success = cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        
        if not success:
            raise ValueError("JPEG save failed")
        
        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(temp_path)
        
    
    return dim, quantization, Y_coefficients, CbCr_coefficients


def extract_overlapping_blocks(img, window_size, crop_size, channels):
    return torch.nn.functional.unfold(img, kernel_size=window_size, stride=crop_size,
                                    padding=0).transpose(1, 2).reshape(-1, channels, window_size, window_size)

def reconstruct_image(output_blocks, num_h, num_w):
    img_E = torch.cat(output_blocks, dim=0)
    img_E = rearrange(img_E, '(h w) c ph pw -> 1 c (h ph) (w pw)', h=num_h, w=num_w)
    return img_E


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
