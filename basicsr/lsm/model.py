import torch
import torch.nn as nn

from basicsr.lsm.context2 import ContextSR
from basicsr.lsm.subspace import Subspace


class Model(nn.Module):

    def __init__(self, levels=None,subspace_dims=None,group_dims=None,scale_factors=None,warm_up_weight=None,coarse2fine_weight=None):
        super(Model, self).__init__()

        self.levels = levels
        self.subspace_dims = subspace_dims
        self.group_dims = group_dims
        self.scale_factors = scale_factors
        self.warm_up_weight = warm_up_weight
        self.coarse2fine_weight = coarse2fine_weight

        self.iters = [2, 2, 2, 2]
        self.block_size = [4, 4, 4, 4]
        self.spp_size = [[3, 5, 7, 9]]
        self.spp_chan = [[2, 1, 1, 1]]

    def minimize_1d(self,diff,im_0,im_0_mean,subspace):
        # input: minimization context, previous step output, current step subspace
        N, _, H, W = im_0.shape
        subspace_dim = subspace.shape[-1]

        diff = torch.reshape(im_0, (N, H * W, 1))
        hessian = torch.matmul(subspace.t(), subspace)
        difference = torch.matmul(subspace.t(), im_0)

        if im_0 is not None:
            im_0_reshaped = torch.reshape(im_0 - im_0_mean, (N, H * W, 1))
            # the c of previous step output in current subspace


