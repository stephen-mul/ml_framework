train_params = {
    'num_channels': 1,
    'num_classes': 10,
    'n_epochs': 100,
    'dataset': 'MNIST',
    'batch_size': 32,
    'optimiser': 'ADAM', 
    'learning_rate': 1e-2,
    'scheduler': 'simple_decay',
    'loss': 'cross_entropy'
}

test_params = {
    'num_channels': 1,
    'num_classes': 10,
    'dataset': 'MNIST'
}