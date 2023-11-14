import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
import config
from conv_vae import VAE
from discriminator import discriminator
from one_hot_encoder import ohe
from losses import new_vae_loss, cross_entropy
from network_utils import EarlyStop, binary, normalise
from torch.utils.data import DataLoader
from custom_dataloader.custom_elv import customDataset
from custom_dataloader.augmentations import RotateTransform
from torchsummary import summary

def main():
    DATASET = config.dataset
    summary_mode = config.summary
    scheduler_type = config.lr_scheduler

    #################
    ### Load Data ###
    #################

    if DATASET == 'MNIST':

        mnist_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: binary(x))
        ])

        train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=mnist_transform)
        train_iter = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=torch.get_num_threads())
    elif DATASET == 'custom':
        #processed_path = './data/test'
        processed_path = './data/random_tile_200'
        train_iter = DataLoader(customDataset(processed_path, transform=RotateTransform([0, 90, 180, 270])), batch_size = 32,
                                    shuffle = True, num_workers=torch.get_num_threads())

    ##################
    ### Load Model ###
    ##################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DATASET == 'MNIST':
        net = discriminator()
        encoder = ohe()
    elif DATASET == 'custom':
        net = VAE((1, 32, 32), nhid = 16, elv=True)
    net.to(device)

    if summary_mode and DATASET == 'custom':
        summary(net, (1, 32, 32))
        exit()
    elif summary_mode and DATASET == 'MNIST':
        summary(net, (1, 28,28))
        exit()
        
    weight_dir = './weights/new_model/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        print(f'Created new folder: {weight_dir}')
    save_name = f'{weight_dir}VAE.pt'


    ################
    ### Training ###
    ################

    lr = 1e-2
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

    def adjust_lr(optimiser, decay_rate = 0.95):
        for param_group in optimiser.param_groups:
            param_group['lr'] *= decay_rate

    #################
    ### Cyclic LR ###
    #################
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr = 1e-3, max_lr = 1e-1, cycle_momentum=False, 
                                                    step_size_up=20, step_size_down=20)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    retrain = True
    if os.path.exists(save_name):
        print("Model parameters have already been trained. Retrain ? [y/n]")
        #ans = input()
        ans = 'y'
        if not (ans == 'y'):
            checkpoint = torch.load(save_name, map_location=device)
            net.load_state_dict(checkpoint["net"])
            optimiser.load_state_dict(checkpoint["optimiser"])
            for g in optimiser.param_groups:
                g['lr'] = lr

    
    early_stop = EarlyStop(patience = 20, save_name = save_name)
    net = net.to(device)
    
    max_epochs = config.nepochs
    print('Training on ', device)
    for epoch in range(max_epochs):
        train_loss, n , start = 0.0, 0, time.time()
        if DATASET == 'MNIST':
            for X, y in tqdm.tqdm(train_iter, ncols = 50):
                X = X.to(device)
                y = encoder.encode(y).to(device)
                y_hat = net(X)

                l = cross_entropy(y_hat, y).to(device)
                optimiser.zero_grad()
                l.backward()
                optimiser.step()

                train_loss += l.cpu().item()
                n += X.shape[0]
            
            train_loss /= n
            lr = get_lr(optimiser)

            #print('epoch %d, train loss %.4f , time %.1f sec'
            #% (epoch, train_loss, time.time() - start))
            print(f'epoch {epoch}, train loss {train_loss}, time {round(time.time() -start, 1)} sec, lr {round(lr, 4)}')
        
            if scheduler_type == 'simple_decay':
                adjust_lr(optimiser)
            elif scheduler_type == 'cyclic':
                scheduler.step()
            else:
                print('Select a valid lr scheduler - simple_decay or cyclic')
                exit()
            
            if (early_stop(train_loss, net, optimiser)):
                break
        elif DATASET == 'custom':
            for batch in tqdm.tqdm(train_iter, ncols = 50):
                im0 = normalise(batch['image_0'].to(device))
                im1 = normalise(batch['image_1'].to(device))
                #print('Input shape: ', im0.shape)
                #print('Target shape:', im1.shape)
                im1_hat, mean, logvar = net(im0)
                #print('Out shape: ', im1_hat.shape)

                l = new_vae_loss(im1, im1_hat, mean, logvar).to(device)
                optimiser.zero_grad()
                l.backward()
                optimiser.step()

                train_loss += l.cpu().item()
                n += im0.shape[0]
            
            train_loss /= n
            lr = get_lr(optimiser)

            #print('epoch %d, train loss %.4f , time %.1f sec'
            #% (epoch, train_loss, time.time() - start))
            print(f'epoch {epoch}, train loss {round(train_loss, 4)}, time {round(time.time() -start, 1)} sec, lr {round(lr, 4)}')
        
            if scheduler_type == 'simple_decay':
                adjust_lr(optimiser)
            elif scheduler_type == 'cyclic':
                scheduler.step()
            else:
                print('Select a valid lr scheduler - simple_decay or cyclic')
                exit()
            
            if (early_stop(train_loss, net, optimiser)):
                break

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])




if __name__ == "__main__":
    main()