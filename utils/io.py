"""Utility functions for Input/Output."""

import os
from datasets import power,gas,hepmass,miniboone,bsds300,moons
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as T

class NoDataRootError(Exception):
    """Exception to be thrown when data root doesn't exist."""
    pass


def get_data_root():
    """Returns the data root, which we assume is contained in an environment variable.

    Returns:
        string, the data root.

    Raises:
        NoDataRootError: If environment variable doesn't exist.
    """
    data_root_var = 'DATAROOT' 
    try:
        return os.environ[data_root_var]
    except KeyError:
        print('default data path: ' + './data')
        return './data'
        #raise NoDataRootError('Data root must be in environment variable {}, which'
        #                      ' doesn\'t exist.'.format(data_root_var))

def train_transforms(dataset):
    if dataset == "cifar10":
        return T.Compose(
            [
            T.Resize(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
        )
    elif dataset == "celeba":
        return T.Compose(
            [
                T.CenterCrop(148),
                T.Resize(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
    elif dataset == "mnist" or dataset == "fmnist":
        return T.Compose([T.Resize(32),T.Grayscale(3),T.ToTensor()])


def get_data(data):
    if data=='power':
        trainset = power.PowerDataset(split='train')
        valset = power.PowerDataset(split='val')
        testset = power.PowerDataset(split='test')
        data_type = "tabular"
    elif data=='hepmass':
        trainset = hepmass.HEPMASSDataset(split='train')
        valset = hepmass.HEPMASSDataset(split='val')
        testset = hepmass.HEPMASSDataset(split='test')
        data_type = "tabular"
    elif data=='miniboone':
        trainset = miniboone.MiniBooNEDataset(split='train')
        valset = miniboone.MiniBooNEDataset(split='val')
        testset = miniboone.MiniBooNEDataset(split='test')
        data_type = "tabular"
    elif data=='gas':
        trainset = gas.GasDataset(split='train')
        valset = gas.GasDataset(split='val')
        testset = gas.GasDataset(split='test')
        data_type = "tabular"
    elif data=='bsds300':
        trainset = bsds300.BSDS300Dataset(split='train')
        valset = bsds300.BSDS300Dataset(split='val')
        testset = bsds300.BSDS300Dataset(split='test')
        data_type = "tabular"
    elif data=='moons':
        trainset = moons.MoonDataset(split='train')
        valset = moons.MoonDataset(split='val')
        testset = moons.MoonDataset(split='test')
        data_type = "toy"
    elif data == "mnist":
        trainset = datasets.MNIST(get_data_root(), transform=train_transforms(data), train=True, download=True)
        valset = datasets.MNIST(get_data_root(), transform=train_transforms(data), train=False, download=True)
        testset = datasets.MNIST(get_data_root(), transform=train_transforms(data), train=False, download=True)
        data_type = "vision"
    elif data == "fmnist":
        trainset = datasets.FashionMNIST(get_data_root(), transform=train_transforms(data), train=True, download=True)
        valset = datasets.FashionMNIST(get_data_root(), transform=train_transforms(data), train=False, download=True)
        testset = datasets.FashionMNIST(get_data_root(), transform=train_transforms(data), train=False, download=True)
        data_type = "vision"
    elif data == "cifar10":
        trainset = datasets.CIFAR10(get_data_root(), transform=train_transforms(data), train=True, download=True)
        valset = datasets.CIFAR10(get_data_root(), transform=train_transforms(data), train=False, download=True)
        testset = datasets.CIFAR10(get_data_root(), transform=train_transforms(data), train=False, download=True)
        data_type = "vision"
    elif data == "celeba":
        trainset = datasets.CelebA(get_data_root(), transform=train_transforms(data), split='train')
        valset = datasets.CelebA(get_data_root(), transform=train_transforms(data), split='valid')
        testset = datasets.CelebA(get_data_root(), transform=train_transforms(data), split='test')
        data_type = "vision"
    else:
        raise
    return trainset, valset, testset, data_type


def savefig_toy(flow,K, i=0, device = "cuda"):
    fig, ax = plt.subplots(1, K)
    xline = torch.linspace(-2, 2,100)
    yline = torch.linspace(-1, 1,100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1).to(device)
    zs = []
    zgrids = []
    with torch.no_grad():
        for k in range(K):
            zs.append(torch.zeros(10000, K).to(device))
            zs[-1][:,k]=1

            zgrids.append(flow.log_prob(xyinput, zs[-1]).exp().reshape(100, 100).cpu())
        
    for k in range(K):
        ax[k].contourf(xgrid.numpy(), ygrid.numpy(), zgrids[k].numpy())
    
    plt.title('iteration {}'.format(i + 1))
    
    
    plt.savefig(f"./output/training_{str(i + 1).zfill(5)}_image.png")
    plt.close()