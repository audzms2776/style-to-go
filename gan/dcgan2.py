import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

batch_size = 64
noise_size = 100
total_epoch = 100
img_size = 28
output_channel = 1


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_size = (img_size // 4)

        self.l1 = nn.Sequential(nn.Linear(noise_size, 128 * self.init_size ** 2))
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channel, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)

        return img


def disc_block(in_filters, out_filters, bn=True):
    block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Dropout2d(0.25)]

    if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))

    return block


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            *disc_block(output_channel, 16, bn=False),
            *disc_block(16, 32),
            *disc_block(32, 64),
            *disc_block(64, 128),
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)

        adv_layer = nn.Sequential(nn.Linear(out.shape[1], 1),
                                  nn.Sigmoid())

        if torch.cuda.is_available():
            adv_layer.cuda()

        validity = adv_layer(out)

        return validity


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_noise(x_size):
    return torch.randn(x_size, noise_size)


# Image processing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# MNIST dataset
mnist = datasets.MNIST(root='/tmp/',
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

if torch.cuda.is_available():
    D.cuda()
    G.cuda()
    bce_loss.cuda()

for epoch in range(total_epoch):
    pbar = tqdm(data_loader)
    pbar.set_description('{}'.format(epoch))

    for x, _ in pbar:
        # input data
        curr_batch = x.size(0)
        x_data = to_var(x)
        sample_noise = to_var(get_noise(curr_batch))

        # label
        one_label = Variable(torch.ones(curr_batch, 1), requires_grad=False).cuda()
        zero_label = Variable(torch.zeros(curr_batch, 1), requires_grad=False).cuda()

        fake_img = G(sample_noise)

        # discriminator

        real_loss = bce_loss(D(x_data), one_label)
        fake_loss = bce_loss(D(fake_img), zero_label)

        d_loss = (real_loss + fake_loss) / 2

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # generator

        g_loss = bce_loss(D(fake_img), one_label)

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        g_loss.backward()
        g_optimizer.step()

    fake_images = fake_img.view(fake_img.size(0), 1, 28, 28)
    save_image(fake_images, '{}.png'.format(epoch), normalize=True)
