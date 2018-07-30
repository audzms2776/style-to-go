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

# def get_noise():
#     return torch.randn(5, 3, 256, 256)
#
#
# M = ConvNet().cuda()
# print(M(to_var(get_noise())).shape)
