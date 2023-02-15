from typing import List

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from src.model.normalizing_flow.classic.modules import *
from src.utils.utils import sampling
import math

from torch.nn import BCELoss


class Flow_Encoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols):
        super(Flow_Encoder, self).__init__()

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


class Flow_Decoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols):
        super(Flow_Decoder, self).__init__()

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


class FlowModel(nn.Module):
    def __init__(
            self,
            flows: List[str],
            hidden_sizes_encoder,
            hidden_sizes_decoder,
            z_dim,
            n_channels,
            n_rows,
            n_cols,
            activation=torch.tanh
    ):
        super().__init__()

        self.encoder = Flow_Encoder(hidden_sizes_encoder, z_dim, n_channels, n_rows, n_cols)
        self.decoder = Flow_Decoder(hidden_sizes_decoder, z_dim, n_channels, n_rows, n_cols)

        self.net = []

        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(z_dim, activation))

        self.net = nn.Sequential(*self.net)

        self.z_dim = z_dim
        self.n_pixels = n_cols * n_rows

    def forward(self, x: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        mu, log_var = self.encoder(torch.flatten(x, start_dim=1))
        z = sampling(mu, log_var)
        var = torch.exp(log_var)

        log_prob_z0 = (-0.5 * torch.log(torch.tensor(2 * math.pi)) - log_var - 0.5 * ((z - mu) / var) ** 2) \
            .sum(dim=1)

        log_det = torch.zeros((x.shape[0],))

        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        # log_prob_zk = self.prior.log_prob(z)
        log_prob_zk = (-0.5 * (torch.log(torch.tensor(2 * math.pi)) + z ** 2)) \
            .sum(dim=1)

        return self.decoder(z), mu, log_var, log_prob_z0, log_prob_zk, log_det

    def loss_function(self, x, y, mu, log_var):
        reconstruction_error = F.binary_cross_entropy(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels),
                                                      reduction='sum')

        KLD = 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - 1 - log_var)

        return reconstruction_error + KLD


def train_flow(model, optimizer, data_train_loader, n_epoch):
    loss_fn = BCELoss()

    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            x_hat, mu, log_var, log_prob_z0, log_prob_zk, log_det = model(data)
            x_hat_flattened = torch.flatten(x_hat, start_dim=1)
            data_flattened = torch.flatten(data, start_dim=1)

            loss = torch.mean(log_prob_z0) + loss_fn(x_hat_flattened.float(), data_flattened.float()) - torch.mean(
                log_prob_zk) - torch.mean(log_det)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return model


def generate_data(flow_model, n_data=5):
    epsilon = torch.randn(n_data, 1, flow_model.z_dim)
    generations = flow_model.decoder(epsilon)
    return generations
