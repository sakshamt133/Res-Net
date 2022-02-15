import torch
from residual_block import Residual
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size, stride, padding, next_in_channels):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(
            Residual(in_channels, out_channels_list[0], kernel_size, stride, padding,
                     identity_downsample=nn.Conv2d(in_channels, 32, (4, 4), (2, 2), (1, 1))),
            Residual(next_in_channels[0], out_channels_list[1], kernel_size, stride, padding,
                     identity_downsample=nn.Conv2d(32, 128, (4, 4), (2, 2), (1, 1))
                     ),
            Residual(next_in_channels[1], out_channels_list[2], kernel_size, stride, padding,
                     identity_downsample=nn.Conv2d(128, 512, (4, 4), (2, 2), (1, 1))
                     ),
            Residual(next_in_channels[2], out_channels_list[3], [(3, 3), (3, 3), (3, 3)], [(1, 1), (1, 1), (1, 1)],
                     [(1, 1), (1, 1), (1, 1)]
        ))

    def forward(self, x):
        out = self.resnet(x)
        print(out.shape)
