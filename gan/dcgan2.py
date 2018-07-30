import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt

batch_size = 100
total_epoch = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.f_conv = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64))

        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(64, 4 ** 2 * 1, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(4),
            nn.PReLU(),
            nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1)))

    def forward(self, x):
        conv0_result = self.conv0(x)  # torch.Size([100, 16, 13, 13])

        out = self.conv1(conv0_result) + conv0_result
        out = self.conv2(out) + out
        out = self.conv3(out) + out
        out = self.conv4(out) + out
        out = self.conv5(out) + out

        conv_result = self.f_conv(out) + conv0_result

        img_result = self.pixel_shuffle(conv_result)

        return img_result


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.linear = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_noise():
    return torch.randn(batch_size, 1, 7, 7)


# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
# MNIST dataset
mnist = datasets.MNIST(root='/tmp/data/',
                       train=True,
                       transform=transform,
                       download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

if torch.cuda.is_available():
    D = Discriminator().cuda()
    G = Generator().cuda()
else:
    D = Discriminator()
    G = Generator()

bce_loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

step = 0
step_arr = []
d_loss_arr = []
g_loss_arr = []

for epoch in range(total_epoch):

    for x, _ in data_loader:
        x_data = to_var(x)
        real_label = to_var(torch.ones(batch_size))
        fake_label = to_var(torch.zeros(batch_size))

        d_loss = bce_loss(D(x_data), real_label) \
                 + bce_loss(D(G(to_var(get_noise()))), fake_label)
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        g_loss = bce_loss(D(G(to_var(get_noise()))), real_label)
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        step += 1

        step_arr.append(step)
        d_loss_arr.append(d_loss)
        g_loss_arr.append(g_loss)

    plt.plot(step_arr, d_loss_arr, linestyle='solid')
    plt.plot(step_arr, g_loss_arr, linestyle='solid')
    plt.show()

    fake_images = G(to_var(get_noise()))
    save_image(fake_images, '{}.png'.format(epoch), normalize=True)
