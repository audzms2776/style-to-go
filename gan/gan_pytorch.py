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

D = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid()
)


def g_layer(in_size, out_size, normalize=True):
    layers = [nn.Linear(in_size, out_size)]

    if normalize:
        layers.append(nn.BatchNorm1d(out_size, 0.8))

    layers.append(nn.LeakyReLU(0.2, inplace=True))

    return layers


G = nn.Sequential(
    *g_layer(noise_size, 128, normalize=False),
    *g_layer(128, 256),
    *g_layer(256, 512),
    *g_layer(512, 1024),
    nn.Linear(1024, 784),
    nn.Tanh()
)

bce_loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

for epoch in range(total_epoch):
    pbar = tqdm(data_loader)
    pbar.set_description('{}'.format(epoch))

    for x, _ in pbar:
        # input data
        curr_batch = x.size(0)
        x = to_var(x)
        x_data = to_var(x.view(curr_batch, -1))
        sample_noise = to_var(get_noise(curr_batch))

        # label
        one_label = Variable(torch.ones(x_data.size(0), 1), requires_grad=False).cuda()
        zero_label = Variable(torch.zeros(x_data.size(0), 1), requires_grad=False).cuda()

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
