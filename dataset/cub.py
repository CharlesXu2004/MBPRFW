import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

ROOT_PATH = '/data/fewshotData/PycharmProjects/CUB_200_2011/'
ROOT_PATH_Image = '/data/fewshotData/PycharmProjects/'

class CUB(Dataset):

    def __init__(self, setname, image_size, aug, transform=None):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.setname = setname
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH_Image, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.image_size = image_size
        self.aug = aug
        self.num_classes = len(self.wnids)
        # jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

        # transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        if transform:
            self.transform = transform
        else:
            if self.aug:
                self.transform_0 = transforms.Compose([
                    transforms.RandomResizedCrop(self.image_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                self.transform_1 = transforms.Compose([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.Resize(84),
                    transforms.Resize(92),
                    # transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                self.transform_test = transforms.Compose([
                    # transforms.Resize(84),
                    transforms.Resize(92),
                    # transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.aug:
            image = self.transform_0(Image.open(path).convert('RGB'))
            image_1 = transforms.RandomResizedCrop(self.image_size)(Image.open(path).convert('RGB'))
            image_1 = transforms.functional.resized_crop(image_1, np.random.randint(28), np.random.randint(28), 56, 56, (84, 84))
            image_1 = self.transform_1(image_1)
            return image,image_1,label
        else:
            if self.setname == "train" or self.setname == 'val':
                image = self.transform(Image.open(path).convert('RGB'))
            else:
                image = self.transform_test(Image.open(path).convert('RGB'))
            return image, label

# if __name__ == '__main__':
#     trainset = CUB('test', 84, True)
#     print(len(trainset))