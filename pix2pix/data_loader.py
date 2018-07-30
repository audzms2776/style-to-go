import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

pre_process = transforms.Compose([
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    normalize
])


class TestDataLoader(data.Dataset):
    def __init__(self):
        self.train_path = '/tmp/cityscapes/train'
        self.train_names = os.listdir(self.train_path)

    def __getitem__(self, index):
        img_name = self.train_names[index]

        origin_im = Image.open('/tmp/cityscapes/train_original/' + img_name)
        origin_img = pre_process(origin_im)

        origin_im2 = Image.open('/tmp/cityscapes/train_seg/' + img_name)
        origin_img2 = pre_process(origin_im2)

        return origin_img2, origin_img

    def __len__(self):
        return len(self.train_names)
