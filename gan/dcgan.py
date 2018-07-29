import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F

batch_size = 100
total_epoch = 100


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_noise():
    return torch.randn(batch_size, 1, 10, 10)


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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(12 * 12 * 128, 1)

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([100, 16, 24, 24])
        out = self.conv2(out)  # torch.Size([100, 32, 20, 20])
        out = self.conv3(out)  # torch.Size([100, 64, 16, 16])
        out = self.conv4(out)  # torch.Size([100, 128, 12, 12])
        out = out.view(out.size(0), -1)  # torch.Size([100, 18432])
        out = F.sigmoid(self.fc1(out))  # torch.Size([100, 1])
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=5),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([100, 16, 13, 13])
        out = self.conv2(out)  # torch.Size([100, 32, 16, 16])
        out = self.conv3(out)  # torch.Size([100, 64, 19, 19])
        out = self.conv4(out)  # torch.Size([100, 1, 28, 28])
        return out


if torch.cuda.is_available():
    D = Discriminator().cuda()
    G = Generator().cuda()
else:
    D = Discriminator()
    G = Generator()

bce_loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

for epoch in range(total_epoch):
    pbar = tqdm(data_loader)
    pbar.set_description('{}'.format(epoch))

    for x, _ in pbar:
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

    fake_images = G(to_var(get_noise()))
    save_image(fake_images, '{}.png'.format(epoch), normalize=True)
