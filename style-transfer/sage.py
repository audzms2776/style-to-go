import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def train(args):
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")

    def load_image(image_path, transform=None, max_size=None, shape=None):
        """Load an image and convert it to a torch tensor."""
        image = Image.open(image_path)

        if max_size:
            scale = max_size / max(image.size)
            size = np.array(image.size) * scale
            image = image.resize(size.astype(int), Image.ANTIALIAS)

        if shape:
            image = image.resize(shape, Image.LANCZOS)

        if transform:
            image = transform(image).unsqueeze(0)

        return image.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

    content = load_image(os.path.join(args.data_dir, 'content.jpg'), transform, shape=(200, 200))
    style = load_image(os.path.join(args.data_dir, 'style.jpg'), transform, shape=(200, 200))
    target = content.clone().requires_grad_(True)

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])
    model = VGGNet().to(device).eval()

    for step in range(1, args.steps + 1):
        target_feature = model(target)
        content_feature = model(content)
        style_feature = model(style)

        style_loss = 0
        content_loss = 0

        for f1, f2, f3 in zip(target_feature, content_feature, style_feature):
            content_loss += torch.mean((f1 - f2) ** 2)

            _, c, h, w = f1.size()

            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)

        loss = content_loss + style_loss * args.style_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                        .format(step + 1, args.steps, content_loss.item(), style_loss.item()))

        if step % args.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            img_name = os.path.join(args.model_dir, 'output-{}.png'.format(step))
            torchvision.utils.save_image(img, img_name)

    save_model(model, args.model_dir)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(VGGNet())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--steps', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--style_weight', type=int, default=5)
    parser.add_argument('--sample_step', type=int, default=500)

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
