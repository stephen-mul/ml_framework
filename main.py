import os
import argparse
import config
import dataloaders
import optimisers
import torchvision
from models.classifier import classifier
from losses import cross_entropy
from network_utils import binary

def main(args):
    #### Unpack arguments ####
    mode = args.mode
    name = args.name

    if mode == 'train':
        params = config.train_params
    elif mode =='test':
        params = config.test_params
    else:
        print('Select valid mode: train or test')
        exit()

    ### Define network ###
    net = classifier(img_channels=params['num_channels'], 
                     n_classes=params['num_classes'])

    ### Get Dataloader ###
    if params['dataset'] == 'MNIST':
        mnist_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: binary(x))
        ])
        dataloader = dataloaders.mnist_loader(batch_size=params['batch_size'],
                                              transforms=mnist_transform,
                                              mode=mode)
    else:
        print('Select a valid dataset: MNIST')
        exit()

    ### Get Optimiser ###

    if params['optimiser'] == 'ADAM':
        optimiser = optimisers.adam_optimiser(network=net,
                                              learning_rate=params['learning_rate'],
                                              scheduler=params['scheduler'])
    else:
        print('Select a valid optimiser: ADAM')
        exit()

    ### Get loss function ###
    if params['loss'] == 'cross_entropy':
        loss = cross_entropy
    
    ### Training Loop ###
    net.train(n_epochs=params['n_epochs'], dataloader=dataloader, 
              optimiser=optimiser, loss=loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--name', type=str, default='test_run')
    args = parser.parse_args()
    main(args)