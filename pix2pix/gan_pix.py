import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from data_loader import TestDataLoader
from model import ConvGenNet, ConvDisNet


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def calc_gan_loss(pred_value, flag):
    func_flag = {True: torch.ones_like, False: torch.zeros_like}
    return nn.BCELoss()(pred_value, func_flag[flag](pred_value))


batch_size = 3
learning_rate = 0.001
total_epoch = 1000

custom_dataset = TestDataLoader()
data_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# writer = SummaryWriter()

if torch.cuda.is_available():
    D = ConvDisNet().cuda()
    G = ConvGenNet().cuda()
else:
    D = ConvDisNet()
    G = ConvGenNet()


pixel_loss = nn.L1Loss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

step = 0
step_arr = []
d_loss_arr = []
g_loss_arr = []

for epoch in range(total_epoch):
    for inputs, outputs in data_loader:
        step += 1

        x_data = to_var(inputs)
        y_data = to_var(outputs)

        # D train
        D.zero_grad()
        
        fake_gen = G(x_data)
        pred_fake = D(x_data, fake_gen)
        loss_fake_d = calc_gan_loss(pred_fake, False)

        pred_label = D(x_data, y_data)
        loss_real_d = calc_gan_loss(pred_label, True)

        loss_D = (loss_fake_d + loss_real_d) * 0.5
        loss_D.backward(retain_graph=True)
        d_optimizer.step()

        # G train
        G.zero_grad()

        pred_fake = D(x_data, fake_gen)
        loss_fake_g = calc_gan_loss(pred_fake, True)

        loss_l1_g = pixel_loss(fake_gen, y_data)

        loss_G = loss_fake_g + loss_l1_g * 100
        loss_G.backward()
        g_optimizer.step()

        # logger
        # writer.add_scalar('data/g_loss', loss_G.item(), step)
        # writer.add_scalar('data/d_loss', loss_D.item(), step)

        # grid_img = vutils.make_grid(fake_gen, normalize=True, scale_each=True)
        # writer.add_image('Image', grid_img, step)

        print('D loss: {}, G loss: {}'.format(loss_D, loss_G))
        
    break







