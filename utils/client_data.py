# from mnist.example import x_test as mnist_x_test
# from mnist.example import y_test as mnist_y_test

from mnist_util import load_mnist_data
import torch
import torchvision.datasets as tvds
import numpy as np
import os
import torchvision.transforms as transforms


def get_data(start_batch, batch_size):
    """Get data from mnist and return processed data"""
    (x_train, y_train, x_test, y_test) = load_mnist_data(
        start_batch, batch_size)

    is_test = False
    if is_test:
        data = x_test
        y_test = [y_test]
    else:
        data = x_test.flatten("C")
        # print('data (x_test): ', data)
        # print('y_test: ', y_test)
    data = data.reshape((-1, 28, 28, 1))
    return data, y_test


def get_queries(dataset, subset_indices, batch_size):
    queries = torch.utils.data.Subset(dataset, indices=subset_indices)
    loader = torch.utils.data.DataLoader(queries, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    return loader


def get_dataset(dataset_name):
    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        ds = getattr(tvds, dataset_name.upper())
    else:
        raise ValueError(f'ds: {dataset_name} not supported.')

    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        if dataset_name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset_name == 'cifar100':
            pass
            # transform = # TODO: finish
        elif dataset_name == 'mnist':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.13251461,), (0.31048025,))])
            # transform = # TODO: finish
        trainset = ds(root='./data', train=True, download=True,
                      transform=transform)
        testset = ds(root='./data', train=False, download=True,
                     transform=transform)

    else:
        raise ValueError(f'dataset: {dataset_name} not yet supported.')

    return trainset, testset


def load_data(data_dir):
    fs = os.listdir(data_dir)
    query = None
    labels = None
    noisy = None
    for f in fs:
        if f.find('samples') != -1 and f.find('raw-samples') == -1:
            query = os.path.join(data_dir, f)
        elif f.find('labels') != -1 and f.find('aggregated-labels') == -1:
            labels = os.path.join(data_dir, f)
        elif f.find('aggregated-labels') != -1:
            noisy = os.path.join(data_dir, f)
    if query is None:
        raise ValueError(f'Query file not found in data dir: {data_dir}')
    elif labels is None:
        raise ValueError(f'Labels file not found in data dir: {data_dir}')
    elif noisy is None:
        raise ValueError(f'Noisy labels file not found in data dir: {data_dir}')
    return np.load(query), np.load(labels), np.load(noisy)
