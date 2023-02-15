from typing import List

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from src.model.normalizing_flow.classic.modules import *
import math

from torch.nn import BCEWithLogitsLoss


class FCNEncoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()

        hidden_sizes = [dim_input] + hidden_sizes

        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class FlowModel(nn.Module):
    def __init__(self, flows: List[str], D: int, activation=torch.tanh):
        super().__init__()

        self.prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.net = []

        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(D, activation))

        self.net = nn.Sequential(*self.net)

        self.D = D

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        sigma = torch.exp(log_sigma)
        batch_size = mu.shape[0]
        samples = self.prior.sample(torch.Size([batch_size]))
        z = samples * sigma + mu

        log_prob_z0 = torch.sum(
            -0.5 * torch.log(torch.tensor(2 * math.pi)) -
            log_sigma - 0.5 * ((z - mu) / sigma) ** 2,
            axis=1)

        log_det = torch.zeros((batch_size,))

        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        # log_prob_zk = self.prior.log_prob(z)
        log_prob_zk = torch.sum(
            -0.5 * (torch.log(torch.tensor(2 * math.pi)) + z ** 2),
            axis=1)

        return z, log_prob_z0, log_prob_zk, log_det


class FCNDecoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()

        hidden_sizes = [dim_input] + hidden_sizes
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

    def forward(self, z: torch.Tensor):
        return self.net(z)


def train_flow(encoder, decoder, flow_model, optimizer, data_train_loader, n_epoch):
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            out = encoder(data.view(-1, 784).float())
            mu, log_sigma = out[:, :40], out[:, 40:]
            z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
            x_hat = decoder(z_k)

            loss = torch.mean(log_prob_z0) + loss_fn(x_hat, data.view(-1, 784).float()) - torch.mean(
                log_prob_zk) - torch.mean(log_det)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return encoder, decoder, flow_model


def generate_data(flow_model, decoder, n_data, z_dim):
    mu, log_sigma = torch.randn(n_data, z_dim), torch.randn(n_data, z_dim)
    z, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
    return decoder(z).view(-1, 1, 28, 28)
