import torch
from ResNet import ResNet


in_channels = 3
out_channels_list = [[8, 16, 32], [64, 64, 128], [128, 256, 512], [512, 512, 512]]
next_in_channels = [32, 128, 512, ]
kernel_size_list = [(3, 3), (3, 3), (3, 3)]
stride_list = [(2, 2), (1, 1), (1, 1)]
padding_list = [(1, 1), (1, 1), (1, 1)]


resnet = ResNet(in_channels, out_channels_list, kernel_size_list, stride_list, padding_list, next_in_channels)

x = torch.randn((1, 3, 256, 256))
resnet(x)