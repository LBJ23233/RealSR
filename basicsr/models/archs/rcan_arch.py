import torch
from torch import nn as nn
from matplotlib import pyplot as plt

from basicsr.models.archs.arch_util import Upsample, make_layer


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16, groups=1):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0, groups=groups),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1, groups=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups),
            ChannelAttention(num_feat, squeeze_factor, groups=groups))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1, groups=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB,
            num_block,
            num_feat=num_feat,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale,
            groups=groups)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 groups=1):
        super(RCAN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, groups=groups)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale,
            groups=groups)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups)
        self.upsample = Upsample(upscale, num_feat, groups=groups)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, groups=groups)

    def forward(self, x):
        # self.mean = self.mean.type_as(x)

        # x = (x - self.mean) * self.img_range
        # x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x
        x = self.conv_last(self.upsample(res))
        # feature map
        # f = res.squeeze(0).cpu().numpy()
        # c, h, w = f.shape
        # import os
        # sv_dir = '/home/yangyuqiang/tmp/BasicSR/results/feature_map/'
        # dir_name = 'syn_g10'
        # sv_dir += dir_name
        # if not os.path.exists(sv_dir):
        #     os.makedirs(sv_dir)
        # for i in range(c):
        #     fm = f[i]
        #     plt.imshow(fm)
        #     plt.title(f'{dir_name} c{i}')  # RAW | RGB
        #     plt.savefig(os.path.join(sv_dir, f'c{i}.png'))
        #     plt.close()
        # x = x / self.img_range + self.mean
        # x = x.reshape(-1, 4, x.shape[-2], x.shape[-1])
        return x
