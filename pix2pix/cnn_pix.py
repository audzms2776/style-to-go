import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from data_loader import TestDataLoader
from model import ConvNet


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


batch_size = 128
learning_rate = 0.001
total_epoch = 1000

custom_dataset = TestDataLoader()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
if torch.cuda.is_available():
    model = ConvNet().cuda()
else:
    model = ConvNet()

# if 'model.ckpt' in os.listdir('./'):
#     model.load_state_dict(torch.load('model.ckpt'))

pixel_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(total_epoch):
    for idx, (x, y) in enumerate(train_loader):
        x = to_var(x)
        y = to_var(y)

        output = model(x)
        loss = pixel_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(loss)

    # torch.save(model.state_dict(), 'model.ckpt')
    save_image(output.data, '{}-conv-edge.png'.format(epoch), normalize=True)
