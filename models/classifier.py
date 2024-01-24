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
    def __init__(self, img_channels=1, n_classes=10, 
                 early_stop=True, patience=20, delta=0, 
                 save_path='.checkpoints/checkpoint.pth'):
        super(classifier, self).__init__()
        self.n_classes = n_classes
        
        ### Initialise layers ###
        self.Conv1 = nn.Conv2d(img_channels, 16, 3)
        self.FC1 = nn.Linear(16*26*26, n_classes)
        ### Intialise loss history ###
        self.history=[[],[]]
        ### Initialise early stop object ###
        if early_stop:
            self.EarlyStop = EarlyStop(patience=patience, verbose=False, 
                                       delta=delta, save_path=save_path)

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

    def test(self, dataloader, optimiser, loss, device):
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

                test_loss += 1
                n += X.shape[0]
            #print(loss_total)
            epoch_loss = loss_total/n
            self.EarlyStop(val_loss=epoch_loss, model=self, 
                           optimiser=optimiser)
            total_samples += n
            ### Log loss in history ###
            self.history[1].append(epoch_loss)

            print(f'Number of samples {total_samples}, test loss {round(epoch_loss.item(), 6)}, time {round(time.time() -start, 1)} sec')



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
            #print(loss_total)
            epoch_loss = loss_total/n
            ### Log loss in history ###
            self.history[0].append(epoch_loss)
            lr = optimiser.get_lr()

            print(f'epoch {epoch}, train loss {round(epoch_loss.item(), 6)}, time {round(time.time() -start, 1)} sec, lr {round(lr, 4)}')
            
            optimiser.scheduler_step()

    ### We would like to calculate the test/validation loss each epoch###
    def train_test(self, n_epochs, dataloader, optimiser, loss, device):
        train_iter = dataloader[0]
        test_iter = dataloader[1]
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
            #print(loss_total)
            epoch_loss = loss_total/n
            ### Log training loss in history - test is handle by test()###
            self.history[0].append(epoch_loss)
            lr = optimiser.get_lr()

            print(f'epoch {epoch}, train loss {round(epoch_loss.item(), 6)}, time {round(time.time() -start, 1)} sec, lr {round(lr, 4)}')
            ### Caclculate test loss - must pass optimiser for early stopping###
            self.test(dataloader=test_iter, optimiser=optimiser, 
                      loss=loss, device=device)
            optimiser.scheduler_step()


                
