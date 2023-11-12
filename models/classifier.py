import torch
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from network_utils import EarlyStop, binary, normalise

###################################
### Simple discriminator module ###
###################################

class classifier(nn.Module):
    def __init__(self, img_channels=1, n_classes=10):
        super(classifier, self).__init__()
        self.n_classes = n_classes
        
        ### Initialise layers ###
        self.Conv1 = nn.Conv2d(img_channels, 16, 3)
        self.FC1 = nn.Linear(16*26*26, n_classes)

    def encode(self, input):
        length = list(input.shape)[0]
        vecs = torch.zeros(length, self.n_classes, dtype=torch.float16)
        count = 0
        for integer in input:
            vecs[count, integer] = 1.0
            count += 1
        return vecs

    def forward(self, image):
        x = self.Conv1(image)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        output = F.log_softmax(x)

        return output

    def train(self, n_epochs, dataloader, optimiser, loss, device):
        train_iter = dataloader
        for epoch in range(n_epochs):
            train_loss, n, start = 0.0, 0, time.time()
            loss_total = 0
            for X, y in tqdm(train_iter, ncols=50):
                print(f'Type from dataloader {type(X)}')
                X = X.to(device)
                y = y.to(device)
                y = self.encode(y)
                print(f'Type ofinput {type(X)}')
                print(f'Device in train: {device}')
                y_hat = self.forward(X)
                
                l = loss.loss(y_hat, y)
                loss_total += l
                optimiser.zero_grad()
                l.backward()
                optimiser.step()

                train_loss += 1
                n += X.shape[0]
            print(loss_total)
            train_loss /= n
            lr = optimiser.get_lr()

            print(f'epoch {epoch}, train loss {round(train_loss, 4)}, time {round(time.time() -start, 1)} sec, lr {round(lr, 4)}')
            
            optimiser.scheduler_step()


                
