import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list, identity_downsample=None):
        super(Residual, self).__init__()
        self.b1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size_list[0], stride_list[0], padding_list[0])
        self.relu = nn.ReLU()
        self.b2 = nn.Conv2d(out_channels_list[0], out_channels_list[1], kernel_size_list[1], stride_list[1], padding_list[1])
        self.b3 = nn.Conv2d(out_channels_list[1], out_channels_list[2], kernel_size_list[2], stride_list[2], padding_list[2])
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        out1 = self.relu(self.b1(x))
        out = self.relu(self.b2(out1))
        out = self.b3(out)
        if self.identity_downsample:
            identity = self.identity_downsample(x)
        out = identity + out
        out = self.relu(out)
        return out
