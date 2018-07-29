import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

batch_size = 100
noise_size = 64
total_epoch = 100


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_noise():
    return torch.randn(batch_size, noise_size)


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

D = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(noise_size, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()
)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

bce_loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

for epoch in range(total_epoch):
    pbar = tqdm(data_loader)
    pbar.set_description('{}'.format(epoch))

    for x, _ in pbar:
        # input data
        x = to_var(x)
        x_data = to_var(x.view(batch_size, -1))
        sample_noise = to_var(get_noise())

        # label
        one_label = to_var(torch.ones(batch_size))
        zero_label = to_var(torch.zeros(batch_size))

        # discriminator
        fake_img = G(sample_noise)
        d_fake = D(fake_img)
        d_data = D(x_data)

        d_loss = bce_loss(d_data, one_label) + bce_loss(d_fake, zero_label)
        D.zero_grad()
        G.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # generator
        fake_img = G(sample_noise)
        g_loss = bce_loss(D(fake_img), one_label)
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    fake_images = fake_img.view(fake_img.size(0), 1, 28, 28)
    save_image(fake_images, '{}.png'.format(epoch), normalize=True)
