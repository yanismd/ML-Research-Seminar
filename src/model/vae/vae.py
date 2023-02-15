import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.utils as utils


class VAE_Encoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols):
        super(VAE_Encoder, self).__init__()

        self.n_channels = n_channels
        self.n_rows = n_rows
        self.n_cols = n_cols
        input_dim = n_channels * n_rows * n_cols

        hidden_sizes = [input_dim] + hidden_sizes

        self.net = []
        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

        self.mu_net = nn.Linear(hidden_sizes[-1], z_dim)
        self.sigma_net = nn.Linear(hidden_sizes[-1], z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_net(h), self.sigma_net(h)


class VAE_Decoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols):
        super(VAE_Decoder, self).__init__()

        self.n_channels = n_channels
        self.n_rows = n_rows
        self.n_cols = n_cols
        input_dim = n_channels * n_rows * n_cols

        hidden_sizes = [z_dim] + hidden_sizes
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

        self.output_net = nn.Linear(hidden_sizes[-1], input_dim)

    def forward(self, z: torch.Tensor):
        h = self.net(z)
        return F.sigmoid(self.output_net(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)


class VAE(torch.nn.Module):

    def __init__(self, hidden_sizes_encoder, hidden_sizes_decoder, z_dim, n_channels, n_rows, n_cols):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.n_pixels = n_rows * n_cols

        self.encoder = VAE_Encoder(hidden_sizes_encoder, z_dim, n_channels, n_rows, n_cols)
        self.decoder = VAE_Decoder(hidden_sizes_decoder, z_dim, n_channels, n_rows, n_cols)

    def forward(self, x):
        z_mu, z_log_var = self.encoder(torch.flatten(x, start_dim=1))
        z = utils.sampling(z_mu, z_log_var)
        return self.decoder(z), z_mu, z_log_var

    def loss_function(self, x, y, mu, log_var):
        reconstruction_error = F.binary_cross_entropy(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels),
                                                      reduction='sum')

        KLD = 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - 1 - log_var)

        return reconstruction_error + KLD


def train_vae(vae_model, optimizer, data_train_loader, n_epoch):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()

            y, z_mu, z_log_var = vae_model(data)
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return vae_model


def generate_data(vae_model, n_data=5):
    epsilon = torch.randn(n_data, 1, vae_model.z_dim)
    generations = vae_model.decoder(epsilon)
    return generations


def train_vae_inverse_noise(vae_model, optimizer, data_train_loader, n_epoch, noise_mean, noise_std):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a gaussian noise to the data in entry, the goal being to reconstruct it
            noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(noisy_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return vae_model


def restore_noisy_data(vae_model, clean_data_loader, noise_mean, noise_std):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_vae_inverse_lostdata(vae_model, optimizer, data_train_loader, n_epoch, square_size):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_add_square(data, square_size)
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(lost_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return vae_model


def restore_lostdata_data(vae_model, clean_data_loader, square_size):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_add_square(data, square_size)
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_vae_inverse_blur(vae_model, optimizer, data_train_loader, n_epoch, kernel_size, sigma):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_gaussian_blur(data, kernel_size=kernel_size, sigma=sigma)
            # The model is training on noisy data, encoding the sample in the latent space
            y, z_mu, z_log_var = vae_model(lost_data)
            # The goal is then to decode from the latent space to the restored data
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return vae_model


def restore_blur_data(vae_model, clean_data_loader, kernel_size, sigma):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_gaussian_blur(data, kernel_size=kernel_size, sigma=sigma)
        noisy_data_list.append(noisy_data)

        output_data, z_mu, z_log_var = vae_model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list
