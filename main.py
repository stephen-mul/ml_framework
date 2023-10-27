import os
import argparse
import config

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

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--name', type=str, default='test_run')
    args = parser.parse_args()
    main(args)