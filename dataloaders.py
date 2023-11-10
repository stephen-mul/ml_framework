import torch
import torchvision

from torch.utils.data import DataLoader

class mnist_loader:
    def __init__(self, batch_size=32, shuffle=True, transforms=None, mode='train', root='data') -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        if mode =='train':
            train = True
        else:
            train = False

        self.train_data = torchvision.datasets.MNIST(root=root, train=train, 
                                                     download=True, transform=transforms)

    def get_iter(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                          shuffle=self.shuffle, num_workers=torch.get_num_threads())