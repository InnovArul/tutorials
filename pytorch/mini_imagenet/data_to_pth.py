import torch, os, os.path as osp
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

transform = transforms.Compose([ #torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                #torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

print(transform)

dataset_dir = './data_to_pth/4'
train_data_dir = osp.join(dataset_dir, "train") # put path of training dataset
val_data_dir = osp.join(dataset_dir, "val") # put path of validation dataset
test_data_dir = osp.join(dataset_dir, "test") # put path of test dataset

trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=transform)
valset = torchvision.datasets.ImageFolder(root= val_data_dir, transform=transform)
testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)

def tensorify(dataset, filepath):
    print(dataset_dir, dataset.root, osp.basename(dataset_dir), dataset.classes, dataset.class_to_idx, filepath)
    # input()
    _x, _y = [], []
    for x, y in tqdm(dataset):
        _x.append(x)
        _y.append(y)

    X, Y = torch.stack(_x), torch.tensor(_y)
    torch.save({'split': osp.basename(dataset_dir), 
                'x':X, 'y':Y, 
                'classes':trainset.classes}, filepath)

import os.path as osp

tensorify(trainset, osp.join(dataset_dir, 'train.pth'))
tensorify(valset, osp.join(dataset_dir, 'val.pth'))
tensorify(testset, osp.join(dataset_dir, 'test.pth'))