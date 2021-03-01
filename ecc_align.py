import numpy as np
import rawpy
import imageio
import glob
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def luminance_transfer(ref, tar):
    lu1 = np.std(ref) / np.std(tar)
    lu2 = np.mean(ref) - lu1 * np.mean(tar)
    tar_new = lu1 * tar + lu2
    return (tar_new.clip(0., 1.) * 65535).astype(np.uint16)


def extract_bayer_channels(raw):
    if len(raw.shape) > 2: return raw
    # default : RGGB
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_B = raw[1::2, 1::2]
    img = np.stack((ch_R, ch_Gr, ch_Gb, ch_B), axis=-1)
    return img


def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[1] if dim[1] < img.shape[1] else img.shape[1]
    crop_height = dim[0] if dim[0] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def ecc_align(bayer_hr, bayer_lr, crop_size):
    h, w = crop_size
    bayer_hr = center_crop(bayer_hr, (h, w))
    bayer_lr = center_crop(bayer_lr, (h // 4, w // 4))

    bayer_c = bayer_hr[:, :, 0]
    lr_bayer_c = bayer_lr[:, :, 0]
    img_r = cv2.resize(bayer_c, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_t = lr_bayer_c
    if img_t.dtype == np.uint16:
        img_r = (img_r.astype(np.float32) / 65535)
        img_t = (img_t.astype(np.float32) / 65535)

    
    # Find size of image1
    sz = img_r.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(img_r, img_t, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    # print('ECC done!')
    hr, lr, hr_bic = [], [], []
    for c in range(bayer_hr.shape[-1]):
        bayer_cc, lr_bayer_cc = bayer_hr[:, :, c], bayer_lr[:, :, c]
        sz = lr_bayer_cc.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            img_t_aligned = cv2.warpPerspective(lr_bayer_cc, warp_matrix, (sz[1], sz[0]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            img_t_aligned = cv2.warpAffine(lr_bayer_cc, warp_matrix, (sz[1], sz[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        crop_size = h // 2, w // 2
        hr.append(center_crop(bayer_cc, crop_size))
        lr.append(center_crop(img_t_aligned, (crop_size[0] // 4, crop_size[1] // 4)))
    hr_bic = [cv2.resize(x, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC) for x in hr]
    hr, lr, hr_bic = np.stack(hr, axis=-1), np.stack(lr, axis=-1), np.stack(hr_bic, axis=-1)
    # print('Warp done!')
    return hr, lr, hr_bic


def worker(rp, lr_rp, opt):
    save_dir = opt['save_dir']
    save_type = opt['save_type']

    type = rp.split('.')[-1]
    name = rp.split('/')[-1].split('\\')[-1].split('.')[-2]
    if not rp.endswith('.tif') and not rp.endswith('.tiff'):
        raw = rawpy.imread(rp)
        bayer = raw.raw_image
        lr_raw = rawpy.imread(lr_rp)
        lr_bayer = lr_raw.raw_image
        weight = [1040, 1440] if type == 'CR2' else [1280, 1920]
        # Black Level Correction
        black, white = min(raw.black_level_per_channel), max(raw.camera_white_level_per_channel)
        bayer = (((bayer - black) / (white - black)).clip(0., 1.) * 65535).astype(np.uint16)
        black, white = min(lr_raw.black_level_per_channel), max(lr_raw.camera_white_level_per_channel)
        lr_bayer = (((lr_bayer - black) / (white - black)).clip(0., 1.) * 65535).astype(np.uint16)
    else:
        bayer = imageio.imread(rp)
        lr_bayer = imageio.imread(lr_rp)
        weight = [1040, 1440]
    # split channel, align, and lunimance correction
    bayer_3d = extract_bayer_channels(bayer)
    lr_bayer_3d = extract_bayer_channels(lr_bayer)
    hr, lr, hr_bic = ecc_align(bayer_3d, lr_bayer_3d, weight)
    # hr, lr, hr_bic = bayer_3d, lr_bayer_3d, lr_bayer_3d
    lr = luminance_transfer(hr / 65535, lr / 65535)
    # white balance
    hr[:, :, 0] = ((hr[:, :, 0] / 65535 * (raw.camera_whitebalance[0] / raw.camera_whitebalance[1])).clip(0., 1.) * 65535).astype(np.uint16)
    hr[:, :, 3] = ((hr[:, :, 3] / 65535 * (raw.camera_whitebalance[2] / raw.camera_whitebalance[1])).clip(0., 1.) * 65535).astype(np.uint16)
    lr[:, :, 0] = ((lr[:, :, 0] / 65535 * (lr_raw.camera_whitebalance[0] / lr_raw.camera_whitebalance[1])).clip(0., 1.) * 65535).astype(np.uint16)
    lr[:, :, 3] = ((lr[:, :, 3] / 65535 * (lr_raw.camera_whitebalance[2] / lr_raw.camera_whitebalance[1])).clip(0., 1.) * 65535).astype(np.uint16)

    lr_l_bic = cv2.resize(lr, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    save_path = os.path.join(save_dir, 'HR', name + save_type)
    imageio.imsave(save_path, hr)
    save_path = os.path.join(save_dir, 'Bic', name + '_hr' + save_type)
    imageio.imsave(save_path, hr)
    save_path = os.path.join(save_dir, 'Bic', name + '_lr_bic' + save_type)
    imageio.imsave(save_path, lr_l_bic)
    # save_path = os.path.join(save_dir, 'Bic', name + '_lr' + save_type)
    # imageio.imsave(save_path, lr)
    save_path = os.path.join(save_dir, 'LR2', name + save_type)
    imageio.imsave(save_path, lr)

    process_info = f'Processing {name} ...'
    return process_info


if __name__ == '__main__':
    # raw_dir = 'C:/Users/29685/Desktop/Matlab_Proj/RealSR_sample10/RAW'
    # save_dir = 'C:/Users/29685/Desktop/Matlab_Proj/RealSR_sample10/ISP_ECC'
    # raw = rawpy.imread('/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW/DSC_4140.NEF')  # DSC_4062.NEF 0A5A4375.CR2
    # bayer = np.array(raw.raw_image)
    # mmax, mmin = bayer.max(), bayer.min()
    # raw = rawpy.imread(r'D:\A-PycharmProjects\RAW\DSC_4140.dng')
    # mmax, mmin = raw.max(), raw.min()


    # save_type = '.tiff'
    # raw_dir = r'C:\Users\29685\Desktop\sample\RAW'
    # save_dir = r'C:\Users\29685\Desktop\sample\RAW_ECC'
    # if not os.path.exists(save_dir):
    #     os.makedirs(os.path.join(save_dir, 'HR'))
    #     os.makedirs(os.path.join(save_dir, 'Bic'))
    #     os.makedirs(os.path.join(save_dir, 'LR2'))
    # raw_path = glob.glob(os.path.join(raw_dir, '*'))
    # raw_path.sort()
    #
    # opt = {}
    # opt['save_dir'] = save_dir
    # opt['save_type'] = save_type
    # save_dir = r'C:\Users\29685\Desktop\sample\RAW_LC_T'
    # ###
    # for i in range(0, len(raw_path), 2):
    #     hrp = raw_path[i]
    #     lrp = raw_path[i + 1]
    #     # worker(hrp, lrp, opt)
    #     name = hrp.split('/')[-1].split('.')[-2]
    #     lr_raw, raw = rawpy.imread(lrp), rawpy.imread(hrp)
    #     lr_bayer, bayer = lr_raw.raw_image, raw.raw_image
    #     # Black Level Correction
    #     black, white = min(raw.black_level_per_channel), max(raw.camera_white_level_per_channel)
    #     bayer = (((bayer - black) / (white - black)).clip(0., 1.) * 65535).astype(np.uint16)
    #     black, white = min(lr_raw.black_level_per_channel), max(lr_raw.camera_white_level_per_channel)
    #     lr_bayer = (((lr_bayer - black) / (white - black)).clip(0., 1.) * 65535).astype(np.uint16)
    #     bayer_3d = extract_bayer_channels(bayer)
    #     lr_bayer_3d = extract_bayer_channels(lr_bayer)
    #
    #     # lr_l_bic = cv2.resize(lr_l, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #     # save_path = os.path.join(save_dir, 'LR_bic', name + '_lr_bic' + save_type)
    #     # imageio.imsave(save_path, lr_l_bic)
    #     save_path = os.path.join(save_dir, name + '_hr' + save_type)
    #     imageio.imsave(save_path, bayer_3d)
    #     save_path = os.path.join(save_dir, name + save_type)
    #     imageio.imsave(save_path, lr_bayer_3d)


    # multi-process
    raw_dir = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW'
    save_dir = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECCx4_HOMO'
    save_type = '.tiff'
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'HR'))
        os.makedirs(os.path.join(save_dir, 'Bic'))
        os.makedirs(os.path.join(save_dir, 'LR2'))

    raw_path = glob.glob(os.path.join(raw_dir, '*'))
    raw_path.sort()
    sv_path = glob.glob(os.path.join(save_dir, 'HR', '*'))
    sv_path.sort()
    n_thread = 20
    opt = {}
    opt['save_dir'] = save_dir
    opt['save_type'] = save_type
    opt['raw_path'] = raw_path

    all_name = [raw_path[i].split('/')[-1][:-4] for i in range(len(raw_path))]
    sv_name = [p.split('/')[-1][:-5] for p in sv_path]
    names = list(set(all_name[0::4]) ^ set(sv_name))
    names.sort()
    del names[:7]
    pbar = tqdm(total=len(names), unit='image', desc='Align')
    pool = Pool(n_thread)
    for i in range(len(names)):
        rp = raw_path[all_name.index(names[i])]
        lr_rp = raw_path[all_name.index(names[i]) + 3]
        # worker(rp, lr_rp, opt)
        # pbar.update(1)
        pool.apply_async(
            worker, args=(rp, lr_rp, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done. ')

    # for i in tqdm(range(0, len(raw_path), 4)):
    #     rp = raw_path[i]
    #     lr_rp = raw_path[i + 1]
    #     dir_name = rp.split('/')[-3]
    #     name = rp.split('/')[-1].split('\\')[-1].split('.')[-2]
    #     if not rp.endswith('.tif') and not rp.endswith('.tiff'):
    #         raw = rawpy.imread(rp)
    #         bayer = raw.raw_image
    #         lr_raw = rawpy.imread(lr_rp)
    #         lr_bayer = lr_raw.raw_image
    #         weight = [800, 1400] if dir_name == 'Canon' else [1200, 1800]
    #         # Black Level Correction
    #         black, white = min(raw.black_level_per_channel), max(raw.camera_white_level_per_channel)
    #         bayer = (((bayer - black) / (white - black)) * 65535).astype(np.uint16)
    #         black, white = min(lr_raw.black_level_per_channel), max(lr_raw.camera_white_level_per_channel)
    #         lr_bayer = (((lr_bayer - black) / (white - black)) * 65535).astype(np.uint16)
    #     else:  # tiff or RGB
    #         bayer    = imageio.imread(rp)
    #         lr_bayer = imageio.imread(lr_rp)
    #         weight = [1600, 2800]
    #     bayer_3d = extract_bayer_channels(bayer)
    #     lr_bayer_3d = extract_bayer_channels(lr_bayer)
    #
    #     hr, lr, hr_bic = ecc_align(bayer_3d, lr_bayer_3d, weight)
    #     lr_l = luminance_transfer(hr, lr)
    #     save_path = os.path.join(save_dir, 'HR', name + save_type)
    #     imageio.imsave(save_path, hr)
    #     save_path = os.path.join(save_dir, 'Bic', name + '_hr_bic' + save_type)
    #     imageio.imsave(save_path, hr_bic)
    #     save_path = os.path.join(save_dir, 'Bic', name + '_lr_lu' + save_type)
    #     imageio.imsave(save_path, hr_bic)
    #     save_path = os.path.join(save_dir, 'Bic', name + '_lr' + save_type)
    #     imageio.imsave(save_path, lr)
    #     save_path = os.path.join(save_dir, 'LR2', name + save_type)
    #     imageio.imsave(save_path, lr)



