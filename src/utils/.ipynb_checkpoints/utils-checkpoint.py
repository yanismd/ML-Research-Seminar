import math
import torch
import torchvision.transforms as transforms


def sampling(mu, log_var):
    # this function samples a Gaussian distribution,
    # with average (mu) and standard deviation specified (using log_var)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)  # return z sample


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

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).
    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]