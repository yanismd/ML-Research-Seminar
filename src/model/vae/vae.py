import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, h_dim_1, h_dim_2, z_dim, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = self.n_rows * self.n_cols
        self.z_dim = z_dim

        input_size = n_channels * n_rows * n_cols

        # encoder part
        self.fc1 = nn.Linear(input_size, h_dim_1)
        self.fc2 = nn.Linear(h_dim_1, h_dim_2)
        self.fc31 = nn.Linear(h_dim_2, z_dim)
        self.fc32 = nn.Linear(h_dim_2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim_1)
        self.fc5 = nn.Linear(h_dim_1, h_dim_2)
        self.fc6 = nn.Linear(h_dim_2, input_size)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        # Outputs of the decoder being mu and sigma
        return self.fc31(h), self.fc32(h)

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)

    def sampling(self, mu, log_var):
        # this function samples a Gaussian distribution,
        # with average (mu) and standard deviation specified (using log_var)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        z_mu, z_log_var = self.encoder(torch.flatten(x, start_dim=1))
        z = self.sampling(z_mu, z_log_var)
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

            y, z_mu, z_log_var = vae_model(data)  # FILL IN STUDENT
            loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)  # FILL IN STUDENT
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_train_loader.dataset),
                           100. * batch_idx / len(data_train_loader), loss_vae.item() / len(data)))
        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return vae_model


def generate_data(vae_model, n_data=5):
    epsilon = torch.randn(n_data, 1, vae_model.z_dim)
    generations = vae_model.decoder(epsilon)
    return generations
