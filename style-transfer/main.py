from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


###########

content_name = 'content.jpg'
style_name = 'style.jpg'
total_epoch = 2000
style_weight = 100
sample_step = 500

#########

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])

content = load_image(content_name, transform, shape=(100, 100))
style = load_image(style_name, transform, shape=(100, 100))

target = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([target], lr=1e-3, betas=[0.5, 0.999])
vgg = VGGNet().to(device).eval()

for step in range(total_epoch):
    target_feature = vgg(target)
    content_feature = vgg(content)
    style_feature = vgg(style)

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

    loss = content_loss + style_loss * style_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
          .format(step + 1, total_epoch, content_loss.item(), style_loss.item()))

    if (step + 1) % sample_step == 0:
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
        torchvision.utils.save_image(img, 'output-{}.png'.format(step + 1))
