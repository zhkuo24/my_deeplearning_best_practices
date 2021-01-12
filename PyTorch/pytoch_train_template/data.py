# -*- coding: utf-8 -*-
# @File   : data_loader.py
# @Author : zhkuo
# @Time   : 2020/12/18 10:28
# @Desc   : data set

import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils import *
from config import cfg
from torchvision import datasets, transforms

dataset_mean = [0.4914, 0.4822, 0.4465]
dataset_std = [0.2023, 0.1994, 0.2010]


def get_transforms(phase='train'):
    """
    define the transform for data augmentation
    :param phase: train val or test
    :return: the transform
    """
    assert ('phase' not in ["train", "val", "test"])
    if phase == "train":
        _transforms = A.Compose([
            # A.PadIfNeeded(min_width=40, min_height=40),
            # A.RandomCrop(height=32, width=32),
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.)
        ], p=1.)
    elif phase == "val":
        _transforms = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.)
        ], p=1.)
    elif phase == "test":
        _transforms = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.)
        ], p=1.)
    else:
        _transforms = A.Compose([
            A.Normalize(mean=dataset_mean, std=dataset_mean, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.)
        ], p=1.)

    return _transforms


class CustomDataset(torch.utils.data.Dataset):
    """
    custom dataset according to actual condition
    ! this class is a example, Cannot be used directly
    """
    def __init__(self, datas, labels, transforms=None):
        super(CustomDataset, self).__init__()
        self.train_datas = datas
        self.transforms = transforms
        self.labels = labels

    def __len__(self):
        return len(self.train_datas)

    def __getitem__(self, item):
        img = self.train_datas[item]

        if self.transforms:
            img = self.transforms(image=img)['image']
        label = self.labels[item]
        return img, label


class CatDogDataset(torch.utils.data.Dataset):
    """
    example cats dogs datasets
    """
    def __init__(self, img_names, transforms=None):
        super(CatDogDataset, self).__init__()
        self.img_names = img_names
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        self.read_imgs()

    def read_imgs(self):
        for img_name in self.img_names:
            fname = os.path.basename(img_name)
            if fname.split('.')[0] == 'cat':
                label = 0
            else:
                label = 1
            self.imgs.append(img_name)
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = cv2_imread(img_path)
        if transforms:
            img = self.transforms(image=img)['image']
        label = self.labels[item]
        return img, label


# example
# If you not have the datasets, change download from False to True
if False:
    trainset = datasets.CIFAR10(root='./datas', train=True, download=True, transform=transforms.Compose([
                                                                            transforms.RandomCrop(32, padding=4),
                                                                            transforms.RandomHorizontalFlip(),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                                 (0.2023, 0.1994, 0.2010)),
                                                                            ]))

    testset = datasets.CIFAR10(root='./datas', train=False, download=True, transform=transforms.Compose([
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                                 (0.2023, 0.1994, 0.2010)),
                                                                            ]))


if __name__ == '__main__':

    print(len(trainset), len(testset))
