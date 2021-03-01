import cv2
import glob
import numpy as np
import os.path as osp
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    # Configurations
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # -------------------------------------------------------------------------
    folder_gt = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECC_0217/RGB400/test/HR'
    folder_restored = '/home/yangyuqiang/tmp/BasicSR/results/RGB_cker2real/'
    # crop_border = 4
    suffix = '_s6_50'
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    # img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
        img_restored = cv2.imread(
            osp.join(folder_restored, basename + suffix + ext),
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored],
                                          bgr2rgb=True,
                                          float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        torch.cuda.empty_cache()
        with torch.no_grad():
            lpips_val = loss_fn_vgg(
                img_restored.unsqueeze(0).cuda(),
                img_gt.unsqueeze(0).cuda()).squeeze()

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
