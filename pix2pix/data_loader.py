import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

pre_process = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    normalize
])

# edges2shoes
# cityscapes
class TestDataLoader(data.Dataset):
    def __init__(self):
        self.dataset_name = 'cityscapes'
        self.train_path = '/tmp/{}/train/'.format(self.dataset_name)
        self.train_names = os.listdir(self.train_path)

    def __getitem__(self, index):
        img_name = self.train_names[index]

        origin_im = Image.open('/tmp/{}/train_input/{}'.format(self.dataset_name, img_name))
        input_img = pre_process(origin_im)

        origin_im2 = Image.open('/tmp/{}/train_output/{}'.format(self.dataset_name, img_name))
        output_img = pre_process(origin_im2)

        return input_img, output_img

    def __len__(self):
        return len(self.train_names)
