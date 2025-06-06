import os.path as osp

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = '/data/fewshotData/PycharmProjects/few-shot/split/'
ROOT_PATH_Image = '/data/fewshotData/PycharmProjects/'


class MiniImageNet(Dataset):
    def __init__(self, setname, image_size, aug, transform = None):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.setname = setname
        data = []
        label = []
        label_eq = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH_Image, 'images', name)
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
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.aug:
            if self.setname == "train":
                image = self.transform_0(Image.open(path).convert('RGB'))
                image_1 = transforms.RandomResizedCrop(self.image_size)(Image.open(path).convert('RGB'))
                image_1 = transforms.functional.resized_crop(image_1, np.random.randint(28), np.random.randint(28), 56, 56, (84, 84))
                image_1 = self.transform_1(image_1)
                return image,image_1,label
            else:
                image = self.transform(Image.open(path).convert('RGB'))
                return image, label
        else:
            if self.setname == "train":
                image = self.transform(Image.open(path).convert('RGB'))
            else:
                image = self.transform_test(Image.open(path).convert('RGB'))
            return image, label
# if __name__ == '__main__':
#     trainset = MiniImageNet('train', 84, True)
#     print(len(trainset))
