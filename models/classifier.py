import torch
import time
import os
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
    
    def checkpoint(self, name, net_optimiser):
        weight_dir = f'./weights/{name}/'
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
            print(f'Created new experiment folder: {weight_dir}')
        save_name = f'{weight_dir}classifier.pt'

        ### Define Early Stop object ###
        early_stop = EarlyStop(patience=20, save_name=save_name)

    def test(self, dataloader, loss, device):
        ### same as train loop but we hardcode n_epochs=1###
        test_iter = dataloader
        for epoch in range(1):
            test_loss, n, start = 0.0, 0, time.time()
            loss_total = 0
            total_samples = 0
            for X, y in tqdm(test_iter, ncols=50):
                X = X.to(device)
                y = y.to(device)
                y = self.encode(y).to(device)
                y_hat = self.forward(X)
                
                l = loss.loss(y_hat, y)
                loss_total += l

                test_loss_loss += 1
                n += X.shape[0]
            print(loss_total)
            test_loss /= n
            total_samples += n

            print(f'Number of samples {total_samples}, test loss {round(test_loss, 4)}, time {round(time.time() -start, 1)} sec')



    def train(self, n_epochs, dataloader, optimiser, loss, device):
        train_iter = dataloader
        for epoch in range(n_epochs):
            train_loss, n, start = 0.0, 0, time.time()
            loss_total = 0
            for X, y in tqdm(train_iter, ncols=50):
                X = X.to(device)
                y = y.to(device)
                y = self.encode(y).to(device)
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


                
