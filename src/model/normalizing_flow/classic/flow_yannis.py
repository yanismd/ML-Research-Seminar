from typing import List

import torch
from torch.nn import BCELoss

from src.model.normalizing_flow.classic.modules import *
import src.utils.utils as utils


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
        #print('x : ',   (torch.min(x)))
        #print('weights : ', torch.sum(torch.isnan(self.net[2].weight)))
        h = self.net(x)
        #print('h : ', h)
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
        #return F.sigmoid(self.output_net(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)
        return (self.output_net(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)


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
        #print(torch.flatten(x, start_dim=1))
        #print('x before flatten : ', x.shape)
        mu, log_var = self.encoder(torch.FloatTensor(torch.flatten(x, start_dim=1)))
        #print(torch.sum(torch.isnan(mu)), torch.sum(torch.isnan(log_var)))
        z = utils.sampling(mu, log_var)

        log_prob_z0 = (-0.5 * torch.log(torch.tensor(2 * torch.pi))
                       - log_var - 0.5 * ((z - mu) / torch.sqrt(torch.exp(log_var))) ** 2) \
            .sum(dim=1)

        log_det = torch.zeros((x.shape[0],))

        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        log_prob_zk = (-0.5 * (torch.log(torch.tensor(2 * torch.pi)) + z**2))\
            .sum(dim=1)

        #return self.decoder(z), mu, log_var
        #print(torch.sum(torch.isnan(z)))
        return self.decoder(z), mu,  log_var,log_prob_z0, log_prob_zk, log_det

    #def loss_function(self, x, y, mu, log_var):
    def loss_function(self, x, y, mu,  log_var, log_prob_z0, log_prob_zk, log_det ):
        #reconstruction_error = F.binary_cross_entropy(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels),
        #                                              reduction='sum')
        #print(y.view(-1, self.n_pixels).min(),  x.view(-1, self.n_pixels).min())
        #reconstruction_error_1 = BCELoss(reduction = 'sum')(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels)) #,
        reconstruction_error = nn.MSELoss(reduction='sum')(y.view(-1, self.n_pixels),  x.view(-1, self.n_pixels))                                                                            
        #print('reconstruction_error 1 :', reconstruction_error_1)
        #print('reconstruction_error:', reconstruction_error.shape)
        KLD = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1. - log_var, dim = 1, keepdim = True )
        #print('KLD:', KLD.shape)
        loss = KLD - log_det +reconstruction_error
        #loss = torch.mean(log_prob_z0) + reconstruction_error - torch.mean(log_prob_zk) - torch.mean(log_det)
        #print('reconstruction_error:', reconstruction_error)
        #print('KLD:', KLD)
        #print('log_det : ', log_det)
        #print('loss : ', loss)

        #return reconstruction_error + KLD
        return loss.mean()


def train_flow(model, optimizer, data_train_loader, n_epoch):
    loss_fn = BCELoss()

    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            #y, z_mu, z_log_var = model(data)
            y, z_mu, log_var, log_prob_z0, log_prob_zk, log_det = model(data)
            
            #loss_vae = model.loss_function(data, y, z_mu, z_log_var)
            loss_vae = model.loss_function(data, y, z_mu, log_var, log_prob_z0, log_prob_zk, log_det )
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return model


def generate_data(flow_model, n_data=5):
    epsilon = torch.randn(n_data, 1, flow_model.z_dim)
    generations = flow_model.decoder(epsilon)
    return generations


def train_flow_inverse_noise(model, optimizer, data_train_loader, n_epoch, noise_mean, noise_std):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a gaussian noise to the data in entry, the goal being to reconstruct it
            noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
            # The model is training on noisy data, encoding the sample in the latent space
            #y, z_mu, z_log_var = model(noisy_data)
            y, z_mu,  log_var,log_prob_z0, log_prob_zk, log_det = model(noisy_data)
            # The goal is then to decode from the latent space to the restored data
            #loss_vae = model.loss_function(data, y, z_mu, z_log_var)
            loss_vae = model.loss_function(data, y, z_mu,  log_var,log_prob_z0, log_prob_zk, log_det)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()
        #print('ok')
        if epoch %10==0 :
            print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return model


def restore_noisy_data(model, clean_data_loader, noise_mean, noise_std):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_noise(data, noise_mean, noise_std)
        noisy_data_list.append(noisy_data)

        output_data, _, _, _, _, _ = model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_flow_inverse_lostdata(model, optimizer, data_train_loader, n_epoch, square_size):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_add_square(data, square_size)
            # The model is training on noisy data, encoding the sample in the latent space
            #y, z_mu,  z_log_var  = model(lost_data)
            y, z_mu,  log_var, log_prob_z0, log_prob_zk, log_det  = model(lost_data)   
            # The goal is then to decode from the latent space to the restored data
            #loss_vae = model.loss_function(data, y, z_mu, z_log_var )
            loss_vae = model.loss_function(data, y, z_mu,  log_var,log_prob_z0, log_prob_zk, log_det )
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return model


def restore_lostdata_data(model, clean_data_loader, square_size):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_add_square(data, square_size)
        noisy_data_list.append(noisy_data)

        #output_data, z_mu, z_log_var = model(noisy_data)
        output_data, _, _, _, _, _ = model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list


def train_flow_inverse_blur(model, optimizer, data_train_loader, n_epoch, kernel_size, sigma):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()
            # We add a square at the middle of the data in entry, the goal being to reconstruct the hidden area
            lost_data = utils.pytorch_gaussian_blur(data, kernel_size=kernel_size, sigma=sigma)
            # The model is training on noisy data, encoding the sample in the latent space
            #y, z_mu, z_log_var = model(lost_data)
            y, z_mu,  log_var,log_prob_z0, log_prob_zk, log_det = model(lost_data)
            # The goal is then to decode from the latent space to the restored data
            #loss_vae = model.loss_function(data, y, z_mu, z_log_var)
            loss_vae = model.loss_function(data, y, z_mu,  log_var,log_prob_z0, log_prob_zk, log_det)
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step()

        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_train_loader.dataset)))

    return model


def restore_blur_data(model, clean_data_loader, kernel_size, sigma):
    target_data_list = []
    noisy_data_list = []
    output_data_list = []

    for batch_idx, (data, _) in enumerate(clean_data_loader):
        target_data_list.append(data)

        noisy_data = utils.pytorch_gaussian_blur(data, kernel_size=kernel_size, sigma=sigma)
        noisy_data_list.append(noisy_data)

        #output_data, z_mu, z_log_var = model(noisy_data)
        output_data, _, _, _, _ , _= model(noisy_data)
        output_data_list.append(output_data)

    return target_data_list, noisy_data_list, output_data_list
