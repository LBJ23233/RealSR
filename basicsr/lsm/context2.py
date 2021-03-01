import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# channel ? pad ? gather/index?
class ContextSR(nn.Module):

    def derivative_2d(self, inputs):
        pass

    def __init__(self, group_dim, pooling_size, pooling_chan, is_training, split=None):
        super(ContextSR, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_chan = [group_dim * x for x in pooling_chan]
        self.is_training = is_training
        self.group_dim = group_dim
        self.conv1x1 = nn.Conv2d(group_dim, group_dim, kernel_size=1, stride=1)
        self.conv1x1_1 = nn.Conv2d(group_dim, (group_dim + 2) // 2 * 4, kernel_size=1, stride=1)
        self.context_conv1x1s = []
        for i in range(len(self.pooling_size)):
            self.context_conv1x1s.append(nn.Conv2d(group_dim, 2 * self.pooling_chan[i], kernel_size=1, stride=1))
        self.image_ctx = None

    def forward(self, feature, inputs):
        if self.image_ctx is None:
            image_ctx = self.conv1x1(feature)
            N, C, H, W = image_ctx.shape
            image_ctx = image_ctx.unsqueeze(1)
            self.image_ctx = image_ctx.repeat(1, 3, 1, 1, 1).reshape((-1, C, H, W))
        inputs = torch.cat([self.image_ctx]+inputs, dim=1)
        N, C, H, W = inputs.shape
        outputs = [self.conv1x1_1(inputs)]

        x_int, y_int = np.meshgrid(np.arange(int(W)), np.arange(int(H)))
        x_int = x_int.flatten()
        y_int = y_int.flatten()

        ## multi-scale coordinates
        index00s = []
        index01s = []
        index10s = []
        index11s = []
        areas = []

        for i in range(len(self.pooling_size)):
            half_size = self.pooling_size[i] // 2
            offset_y0 = np.maximum(y_int - half_size, 0)
            offset_x0 = np.maximum(x_int - half_size, 0)
            offset_y1 = np.minimum(y_int + 1 + half_size, int(H))
            offset_x1 = np.minimum(x_int + 1 + half_size, int(W))

            index00 = (offset_y0 * (W + 1) + offset_x0).astype(np.int32)
            index01 = (offset_y0 * (W + 1) + offset_x1).astype(np.int32)
            index10 = (offset_y1 * (W + 1) + offset_x0).astype(np.int32)
            index11 = (offset_y1 * (W + 1) + offset_x1).astype(np.int32)
            area = np.reshape((offset_x1 - offset_x0) * (offset_y1 - offset_y0), (H * W, 1)).astype(np.float64)

            index00s.append(torch.tensor(index00, dtype=torch.int32))
            index01s.append(torch.tensor(index01, dtype=torch.int32))
            index10s.append(torch.tensor(index10, dtype=torch.int32))
            index11s.append(torch.tensor(index11, dtype=torch.int32))
            areas.append(torch.tensor(area, dtype=torch.float64))
        ## integrated features
        inputs     = inputs.permute(2, 3, 0, 1).type(torch.float64)
        intergral  = F.pad(torch.cumsum(torch.cumsum(inputs, dim=0), dim=1), [1, 1, 0, 0])
        intergral  = intergral.reshape(((H + 1) * (W + 1), -1))

        inputs2    = torch.pow(inputs, 2)
        intergral2 = F.pad(torch.cumsum(torch.cumsum(inputs2, dim=0), dim=1), [1, 1, 0, 0])
        intergral2 = intergral2.reshape(((H + 1) * (W + 1), -1))
        ## extract multi-scale features from intergral features
        for i in range(len(self.pooling_size)):
            output = (intergral[index00s[i]] + intergral[index11s[i]]
                      - intergral[index10s[i]] - intergral[index01s[i]]) / areas[i]
            output = output.permute(2, 3, 0, 1)
            # output  = tf.Print(output,[tf.constant(1),tf.constant(self.pooling_size[i]),tf.reduce_max(output,axis=[0,2,3]),tf.reduce_min(output,axis=[0,2,3])],summarize=N*C)

            # output2 = (torch.select(intergral2, index00s[i]) + torch.select(intergral2, index11s[i])
            #            - torch.select(intergral2, index10s[i]) - torch.select(intergral2, index01s[i])) / areas[i]
            # output2 = tf.reshape(tf.transpose(output2, [1, 0]), [N, C, H, W])
            output2 = (intergral2[index00s[i]] + intergral2[index11s[i]]
                      - intergral2[index10s[i]] - intergral2[index01s[i]]) / areas[i]
            output2 = output2.permute(2, 3, 0, 1)
            # output2 = tf.Print(output2,[tf.constant(2),tf.constant(self.pooling_size[i]),tf.reduce_max(output2,axis=[0,2,3]),tf.reduce_min(output2,axis=[0,2,3])],summarize=N*C)
            output2 = output2 - torch.pow(output, 2)

            output = self.context_conv1x1s[i](torch.cat([output, output2], dim=1).type(torch.float32))
            # output  = projection_shortcut(output,self.pooling_chan[i],stride=1,is_training=self.is_training,data_format='channels_first',reuse_variables=reuse_variables,name='spp_%d'%(self.pooling_size[i]+1))

            outputs.append(output)
        return torch.cat(outputs, dim=1)