import torch
import torch.nn as nn

from basicsr.models.archs.arch_util import ResidualBlockNoBN, make_layer


class Subspace(nn.Module):
    def __init__(self, subspace_dim, auxiliary_dim, filters):
        super(Subspace, self).__init__()
        self.subspace_dim = subspace_dim
        self.auxiliary_dim = auxiliary_dim
        self.filters = filters
        block_list = []
        for i in range(len(self.filters)):
            block_list.append(ResidualBlockNoBN(self.filters[i]))
        self.subspace = nn.Sequential(*block_list)
        self.conv1x1 = nn.Conv2d(self.filters[-1], self.subspace_dim+self.auxiliary_dim, kernel_size=1)

    def forward(self, inputs):
        outputs = self.conv1x1(self.subspace(inputs))
        return outputs
