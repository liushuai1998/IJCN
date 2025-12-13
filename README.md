## Blind JPEG Artifacts Removal  via Inverse JPEG Compression (TCSVT 2025)
Shuai Liu, Binqiang Liu, Qingyu Mao, Jiacong Chen, Fanyang Meng, Yonghong Tian
________

:fire::fire: This repository is the official PyTorch implementation of paper "Blind JPEG Artifacts Removal  via Inverse JPEG Compression".
IJCN achieves **state-of-the-art performance** in **BLIND** JPEG artifacts removal on
- Color Single JPEG images
- Color Double JPEG images
- Real-world JPEG images

## Abstract
>Quantization and chroma downsampling are two primary operations that introduce distortions in the JPEG compression. However, most existing blind methods treat artifacts removal as a direct mapping from compressed images to clean ones. They fail to explicitly model the underlying degradation process or design targeted compensation mechanisms. As a result, these methods can only partially remove compression artifacts and struggle to generalize to diverse or unseen degradation scenarios. In this work, we present a novel perspective that formulates artifacts removal as an approximate inversion of the lossy steps in JPEG. Based on this view, we propose an Inverse JPEG Compression Network (IJCN), which aims to progressively compensate for quantization errors and color distortions. Specifically, we first design a Learnable Offset Guidance Module (LOGM) to approximate inverse quantization by modeling both intra-block and inter-block coefficient correlations for predicting rounding offsets. In addition, we propose a Quantization Table Guidance Module (QTGM) that leverages the quantization tables to guide the reconstruction network in mitigating color distortions. By modeling compensation mechanisms under the guidance of quantization tables, IJCN effectively eliminates artifacts across varying compression levels. Extensive experiments demonstrate that IJCN outperforms existing methods in both quantitative metrics and visual quality.
________


## Environment
First you have to make sure that you have installed all dependencies. To do so, you can create an anaconda environment  using the following command:
```bash
conda env create -f environment.yml
conda activate ijcn
```


## Dataset
The **LIVE1** dataset is already included in the `testsets` directory.  If you need additional datasets such as **Urban100**, **BSDS500**, or **ICB**, please use the download links below:

- [Urban100](https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true)
- [BSDS500](http://web.archive.org/web/20160306133802/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
- [ICB](http://imagecompression.info/test_images/rgb8bit.zip)

These datasets are publicly available for research purposes. Please follow the original authors’ licenses and usage terms when using the data.

## Checkpoints
Download the following pre-trained models of IJCN from [Github Release]() to `model_zoo`.
- Color_IJCN.pth
- Colordouble_IJCN.pth


## Test
After preparing the environment, datasets, and pretrained models, run the following commands to test the models. If you want to test your own real JPEG images, modify the `L_path` in `test_real_IJCN.py` to the path of your images.

- Color Single JPEG images
```bash
python test_color_IJCN.py
```

- Color Double JPEG images
```bash
python test_colordouble_IJCN.py
```

- Real-World JPEG images
```bash
python test_real_IJCN.py
```

## Acknowledgement
This code is built on [FBCNN](https://github.com/jiaxi-jiang/FBCNN), [QGAC](https://gitlab.com/Queuecumber/quantization-guided-ac), and [SwinIR](https://github.com/JingyunLiang/SwinIR). We thank the authors for sharing their codes.

## Contact
For any questions or inquiries, please contact us at liushuai981115@163.com or 2410044032@mails.szu.edu.cn.