import torch
import torch.nn as nn


class PixelConvNet(nn.Module):
    def __init__(self):
        super(PixelConvNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), (2, 2), (2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 4 ** 2 * 3, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(4 ** 2 * 3),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1))
        )

    def forward(self, input_data):
        out = self.conv(input_data)
        return out


class PixelConvResNet(nn.Module):
    def __init__(self):
        super(PixelConvResNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.PReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        self.f_conv = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        )

        self.pixel = nn.Sequential(
            nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(1),
            nn.PReLU(),
            nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1))
        )

    def forward(self, input_data):
        conv0_result = self.conv0(input_data)

        x = self.conv1(conv0_result) + conv0_result
        x = self.conv2(x) + x
        x = self.f_conv(x) + conv0_result

        x = self.pixel(x)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1))
        )

    def forward(self, input_data):
        out = self.conv(input_data)
        return out


class ShufflePoolNet(nn.Module):
    def __init__(self):
        super(ShufflePoolNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 4 ** 2 * 3, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(4 ** 2 * 3),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1))
        )

    def forward(self, input_data):
        out = self.conv(input_data)
        return out


class ConvGenNet(nn.Module):
    def __init__(self):
        super(ConvGenNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, input_data):
        out = self.conv(input_data)
        return out

class ConvDisNet(nn.Module):
    def __init__(self):
        super(ConvDisNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x      

import torch
from torch.autograd import Variable


def get_noise():
    return Variable(torch.randn(5, 3, 128, 128)).cuda()


# M = ConvDisNet().cuda()
# oo = M(get_noise(), get_noise())


# M = ConvGenNet().cuda()
# oo = M(get_noise())
# print(oo.shape)
