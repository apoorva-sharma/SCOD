from torchvision.datasets import CIFAR10, CIFAR100
from .cifar10c import CIFAR10C
from torchvision.datasets import SVHN, ImageFolder
from torch.utils.data import Subset, TensorDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import nn_ood

class Cifar10Data(Subset):
    def __init__(self, split, N=None):
        dataset = CIFAR10
        root = os.path.join(nn_ood.DATASET_FOLDER, "CIFAR10")
        if split == "train":
            args = {'train': True,
                    'download': True}
            target_criterion = lambda x: x < 5
        elif split == "val":
            args = {'train': False,
                    'download': True}
            target_criterion = lambda x: x < 5
        elif split == "ood":
            args = {'train': False,
                    'download': True}
            target_criterion = lambda x: x >= 5
        elif split == "svhn":
            dataset = SVHN
            args = {'split': 'train',
                    'download': True}
            root = os.path.join(nn_ood.DATASET_FOLDER, "SVHN")
            target_criterion = lambda x: np.equal( x, x )
        elif split == "tIN":
            dataset = ImageFolder
            args = {}
            root = os.path.join(nn_ood.DATASET_FOLDER, "tiny-imagenet-200/val")
            target_criterion = lambda x: np.equal( x, x )
        elif split == "lsun":
            dataset = ImageFolder
            args = {}
            root = os.path.join(nn_ood.DATASET_FOLDER, "LSUN")
            target_criterion = lambda x: np.equal( x, x )
        elif split == "cifar100":
            dataset = CIFAR100
            args = {
                'train': False,
                'download': True
            }
            root = os.path.join(nn_ood.DATASET_FOLDER, "CIFAR100")
            target_criterion = lambda x: np.equal( x, x )
        elif "cifar10c" in split:
            tokens = split.split(':')
            if len(tokens) != 3:
                raise ValueError
            dataset = CIFAR10C
            args = {
                'corruption': tokens[1],
                'level': int(tokens[2])
            }
            root = os.path.join(nn_ood.DATASET_FOLDER, "CIFAR-10-C")
            target_criterion = lambda x: x < 5
        else:
            print("datatype not understood")
            raise ValueError
            
        self.cifar = dataset(root=root, **args)
        
        self.normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2471, 0.2435, 0.2616)
        )
        
        if split in {'train', 'val', 'ood'}:
            valid_idx = np.flatnonzero(target_criterion(np.array(self.cifar.targets)))
            print(len(valid_idx))
        elif 'cifar10c' in split:
            valid_idx = np.flatnonzero(target_criterion(np.array(self.cifar.labels)))
        else:
            valid_idx = np.arange(len(self.cifar))
        
        if N is not None:
            valid_idx = np.random.choice(valid_idx, N)
        
        super().__init__(self.cifar, valid_idx)
        
        
        
    def __getitem__(self, i):
        input, target = super(Cifar10Data, self).__getitem__(i)
        target = target % 5
        input = transforms.ToTensor()(input)
        input = self.normalize(input)
        return input, target