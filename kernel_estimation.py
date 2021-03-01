import os
import glob
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from scipy.optimize import lsq_linear
from scipy.ndimage import filters
import imageio
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import measurements, interpolation


"""
KernelGAN, nips2019
"""


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def post_process_k(k, n=40):
    """Eliminate negligible values, and centralize k"""
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    # centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return significant_k


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_center_mask(k_size, penalty_scale):
    """Generate a mask of weights ignore values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=7.5, is_tensor=False)
    mask = 1 - mask / (np.max(mask) * 1.1)  # avoid zero at center
    # mask = mask / np.max(mask)
    # margin = (k_size - center_size) // 2 - 1
    # mask[0:margin, 0:margin] = 0
    return mask


# degradation process for avg kernel
def downsample_x2(im, kernel, weight):
    # im have range [0, 1]
    # First run a correlation (convolution with flipped kernel)
    im_blur = np.zeros_like(im).astype(np.float)
    # for channel in range(np.ndim(im)):
    for channel in range(im.shape[2]):
        im_blur[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    lr_down = weight[0][0] * im_blur[::2, ::2, :] \
              + weight[0][1] * im_blur[::2, 1::2, :] \
              + weight[1][0] * im_blur[1::2, ::2, :] \
              + weight[1][1] * im_blur[1::2, 1::2, :]

    return lr_down


"""
convert from matlab code
"""


def kernel_estimation(hr, lr, kernel_radius=8, scale=2, need_lr=False):
    # hr and lr have range 0, 1
    scale = scale
    kernel_size = 2 * kernel_radius + 1
    hr_shape, lr_shape = hr.shape, lr[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius, :].shape

    [x_grid, y_grid] = np.meshgrid(range(lr_shape[0]), range(lr_shape[1]))
    corners = np.hstack((y_grid.reshape(-1, 1), x_grid.reshape(-1, 1)))
    # shift back to original coordinate
    corners += kernel_radius

    # construct matrix for minimize |C*k-d|
    c = lr_shape[2]
    C = np.zeros(shape=(c * corners.shape[0], kernel_size * kernel_size))
    d = np.zeros(shape=(c * corners.shape[0]))
    weight = [[0.25, 0.25], [0.25, 0.25]]
    for i in range(corners.shape[0]):
        for ic in range(c):
            for y in range(scale):
                for x in range(scale):
                    C[c * i + ic] += weight[y][x] * hr[scale * corners[i][1] - kernel_radius + y:scale * corners[i][
                        1] + kernel_radius + y + 1,
                                                    scale * corners[i][0] - kernel_radius + x:scale * corners[i][
                                                        0] + kernel_radius + x + 1, ic].reshape(-1)
            d[c * i + ic] = lr[corners[i][1], corners[i][0], ic]
    ####### constraints
    # matrix for non negative constraint
    lb = np.zeros(shape=(kernel_size * kernel_size))
    ub = np.inf  #lb + 1
    # mask
    center_mask = create_center_mask(kernel_size, 30)
    C = C * center_mask.reshape((1, -1))

    # solve Ax=b
    """
    matlab lsqlin(C, d, A, b):         minimize 0.5*(NORM(C*x-d)).^2       subject to    A*x <= b
    scipy lsq_linear(A, b, (lb, ub)):  minimize 0.5 * ||A x - b||**2       subject to lb <= x <= ub
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
    """
    res = lsq_linear(C, d, bounds=(lb, ub))
    kernel = res.x.reshape([kernel_size, kernel_size])

    # degradation
    if need_lr:
        lr_est = downsample_x2(hr, kernel, weight)

    return (kernel, lr_est) if need_lr else (kernel, [])


def worker(hrp, lr_dir, save_dir, need_lr=False):
    name = hrp.split('/')[-1][:-5]
    hr = imageio.imread(hrp) / 65535.
    lr = imageio.imread(os.path.join(lr_dir, name + '.tiff')) / 65535.
    kernel, lr_syn = kernel_estimation(hr, lr, need_lr=need_lr)
    kernel = post_process_k(kernel)
    sio.savemat(os.path.join(save_dir, f'{name}_kernel_x2.mat'), {'Kernel': kernel})
    if need_lr:
        imageio.imwrite(os.path.join(lr_dir + 'syn', f'{name}.tiff'), (lr_syn.clip(0., 1.) * 65535).astype(np.uint16))


if __name__ == '__main__':
    # local = False
    local = True
    if not local:
        import torch
        hr_dir = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECC_0217/RAW400WB/x2/train/HRsub'
        lr_dir = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECC_0217/RAW400WB/x2/train/LR2sub'
        save_dir = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECC_0217/RAW400WB/x2/train/kernel_center75'

        # save_lr = '/home/yangyuqiang/datasets/RealSR/RealSR_Raw/RAW_BLC_ECC_0217/RAW400WB/x2/test/LR2syn'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        hr_path = glob.glob(hr_dir + '/*')
        hr_path.sort()
        k_path = glob.glob(save_dir + '/*')
        k_path.sort()
        knames = [kp.split('/')[-1][:-14] for kp in k_path]
        allname = [p.split('/')[-1][:-5] for p in hr_path]
        names = list(set(allname) ^ set(knames))
        names.sort()
        n_thread = 20
        need_lr = False
        pbar = tqdm(total=len(names), unit='image', desc='kernel')
        pool = Pool(n_thread)
        for name in names:
            hrp = os.path.join(hr_dir, name + '.tiff')
            # worker(hrp, lr_dir, save_dir, need_lr)
            pool.apply_async(
                worker, args=(hrp, lr_dir, save_dir, need_lr), callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()
        pbar.close()
        print('All processes done. ')
    else:
        # kernel results visualization
        save_dir = r'Y:\datasets\RealSR\RealSR_Raw\RAW_BLC_ECC_0217\RAW400WB\x2\train\kernel_center75'
        save_img = r'C:\Users\29685\Desktop\Matlab_Proj\RealSR_sample10\real_kernel'
        # save_img = r'Y:\datasets\RealSR\RealSR_Raw\RAW_BLC_ECC_0217\RAW400WB\x2\train\kernel_center4_img'
        if not os.path.exists(save_img):
            os.mkdir(save_img)
        k_path = glob.glob(save_dir + '\\*mat')
        k_path.sort()
        kernels = 0
        cnt = 0
        thres = [20, 60, 200, 500, 1000, 2000]
        for kp in tqdm(k_path):
            name = kp.split('/')[-1].split('\\')[-1][:-14]
            data = sio.loadmat(kp)
            kernel = data['Kernel']
            kernel /= kernel.sum()
            kernels += kernel
            cnt += 1
            # plt.imshow(kernel)
            # plt.colorbar()
            # plt.title(f'{name}')
            # plt.savefig(os.path.join(save_img, f'{name}_center6.png'))
            # plt.close()

            if cnt in thres:
                ker = kernels / cnt
                plt.imshow(ker)
                plt.colorbar()
                plt.title(f'{cnt}')  # RAW | RGB
                plt.savefig(os.path.join(save_img, f'{cnt}_center75.png'))
                plt.close()
            if cnt == 2000: break


        # center_mask = create_center_mask(17, 30)
        # tmp = np.ones(shape=(3, 17 * 17))
        # tmp = tmp * center_mask.reshape((1, -1))
        # tmp = tmp.reshape((3, 17, 17))



