"""
Data loader for all datasets
"""

from torchvision import datasets, transforms
import torch


def fetch_mnist_loader(
        n_samples_train=1000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    mnist_trainset = datasets.MNIST(f'{path_to_data}', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(f'{path_to_data}', train=False, download=True, transform=transforms.ToTensor())

    # create data loader with said dataset size
    mnist_trainset_reduced = torch.utils.data.random_split(
        mnist_trainset,
        [n_samples_train, len(mnist_trainset) - n_samples_train]
    )[0]
    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset_reduced, batch_size=batch_size, shuffle=True,
                                                     drop_last=True)

    # download test dataset
    mnist_testset_reduced = torch.utils.data.random_split(
        mnist_testset,
        [n_samples_test, len(mnist_testset) - n_samples_test]
    )[0]
    mnist_test_loader = torch.utils.data.DataLoader(mnist_testset_reduced, batch_size=batch_size, shuffle=True,
                                                    drop_last=True)

    return mnist_train_loader, mnist_test_loader
