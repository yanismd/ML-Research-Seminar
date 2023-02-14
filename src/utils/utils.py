import torch
import torchvision.transforms as transforms


def pytorch_to_numpy(x):
    return x.detach().numpy()


def pytorch_noise(x, noise_mean, noise_std):
    return x + (noise_mean + noise_std * torch.randn(x.shape))


def pytorch_add_square(x, square_size):
    tensor = x.clone()
    img_width = x.shape[3]
    img_height = x.shape[2]
    tensor[:, :, img_height // 2 - square_size:img_height // 2 + square_size,
    img_width // 2 - square_size:img_width // 2 + square_size] = 0
    return tensor


def pytorch_gaussian_blur(x, kernel_size, sigma):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(x)
