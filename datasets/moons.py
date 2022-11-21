import numpy as np
import os
import utils

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

import sklearn.datasets




def load_moons():
    x, _ = sklearn.datasets.make_moons(1024, noise=.01,random_state=2022)
    x[:,0] -= 0.5
    x[:,1] -= 0.25

    x_te, _ = sklearn.datasets.make_moons(1024, noise=.01,random_state=2023)
    x_te[:,0] -= 0.5
    x_te[:,1] -= 0.25

    x_val, _ = sklearn.datasets.make_moons(1024, noise=.01,random_state=2021)
    x_val[:,0] -= 0.5
    x_val[:,1] -= 0.25

    return x.astype('float32'),x_val.astype('float32'),x_te.astype('float32')





def print_shape_info():
    train, val, test = load_moons()
    print(train.shape, val.shape, test.shape)


class MoonDataset(Dataset):
    def __init__(self, split='train', frac=None):
        train, val, test = load_moons()
        
        self.data = eval(split)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def main():
    dataset = MoonDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    plt.hist(dataset.data.reshape(-1), bins=250)
    plt.show()


if __name__ == '__main__':
    main()
