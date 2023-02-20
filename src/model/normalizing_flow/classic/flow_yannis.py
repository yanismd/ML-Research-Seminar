from typing import List
from torchvision import models
from torchsummary import summary
import torch
from torch.nn import BCELoss

from src.model.normalizing_flow.classic.modules import *
import src.utils.utils as utils


class Flow_Encoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols, conv_sizes, conv_kernel_sizes,conv_strides):
        super(Flow_Encoder, self).__init__()

        self.n_channels = n_channels
        self.n_rows = n_rows
        self.n_cols = n_cols
        input_dim = n_channels * n_rows * n_cols

        #hidden_sizes = [input_dim] + hidden_sizes

        self.fc_net = []
        self.conv_net = []
        for i in range(len(conv_sizes)-1):
            self.conv_net.append( nn.Conv2d(conv_sizes[i], conv_sizes[i + 1], conv_kernel_sizes[i], conv_strides[i] , padding = 'same'))
            self.conv_net.append(nn.ReLU())
        #self.conv_net.append(nn.Flatten())

        for i in range(len(hidden_sizes) - 1):
            self.fc_net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fc_net.append(nn.ReLU())

        
            #self.net.append()
        self.fc_net = nn.Sequential(*self.fc_net)
        self.conv_net = nn.Sequential(*self.conv_net)

        self.mu_net = nn.Linear(hidden_sizes[-1], z_dim)
        self.sigma_net = nn.Linear(hidden_sizes[-1], z_dim)
        #summary(self, ( 3, 32, 32))

    def forward(self, x):
        c = self.conv_net(x)
        c = c.view(c.size(0), -1)

        h = self.fc_net(c)
        return self.mu_net(h), self.sigma_net(h)


class Flow_Decoder(torch.nn.Module):
    def __init__(self, hidden_sizes, z_dim, n_channels, n_rows, n_cols, conv_sizes, conv_kernel_sizes,conv_strides):
        super(Flow_Decoder, self).__init__()

        self.n_channels = n_channels
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.conv_sizes = conv_sizes
        input_dim = n_channels * n_rows * n_cols

        hidden_sizes = [z_dim] + hidden_sizes
        self.conv_net = []
        self.fc_net = []
        for i in range(len(hidden_sizes) - 1):
            self.fc_net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fc_net.append(nn.ReLU())
        
        for i in range(len(conv_sizes)-2):
            self.conv_net.append( nn.ConvTranspose2d(conv_sizes[i], conv_sizes[i + 1], conv_kernel_sizes[i], conv_strides[i] ))
            self.conv_net.append(nn.ReLU())
        self.conv_net.append( nn.Conv2d(conv_sizes[len(conv_sizes)-2], conv_sizes[len(conv_sizes)- 1], conv_kernel_sizes[len(conv_sizes)-2], conv_strides[len(conv_sizes)-2] ))
        self.conv_net.append(nn.Sigmoid())

        self.conv_net = nn.Sequential(*self.conv_net)
        self.fc_net = nn.Sequential(*self.fc_net)

        self.output_net = nn.Linear(conv_sizes[-2], input_dim)

    def forward(self, z: torch.Tensor):
        h = self.fc_net(z)
        c = self.conv_net(h.view(-1, self.n_cols, self.n_rows, self.conv_sizes[0]))
        #return F.sigmoid(self.output_net(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)
        return c


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
            activation=torch.tanh,
            conv_sizes = [],
            conv_kernel_sizes = [],
            conv_strides = [],
    ):
        super().__init__()

        self.encoder = Flow_Encoder(hidden_sizes_encoder, z_dim, n_channels, n_rows, n_cols, conv_sizes, conv_kernel_sizes,conv_strides )
        self.decoder = Flow_Decoder(hidden_sizes_decoder, z_dim, n_channels, n_rows, n_cols, conv_sizes[::-1], conv_kernel_sizes[::-1],conv_strides[::-1])

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

        #mu, log_var = self.encoder(torch.FloatTensor(torch.flatten(x, start_dim=1)))
        mu, log_var = self.encoder(torch.FloatTensor(x))
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

        return self.decoder(z), mu,  log_var,log_prob_z0, log_prob_zk, log_det

    #def loss_function(self, x, y, mu, log_var):
    def loss_function(self, x, y, mu,  log_var, log_prob_z0, log_prob_zk, log_det ):
        #reconstruction_error = F.binary_cross_entropy(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels),
        #                                              reduction='sum')
        #reconstruction_error_1 = BCELoss(reduction = 'sum')(y.view(-1, self.n_pixels), x.view(-1, self.n_pixels)) #,
        reconstruction_error = nn.MSELoss(reduction='sum')(y.view(-1, self.n_pixels),  x.view(-1, self.n_pixels))                                                                            
        KLD = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1. - log_var, dim = 1, keepdim = True )
        loss = KLD - log_det +reconstruction_error
        #loss = torch.mean(log_prob_z0) + reconstruction_error - torch.mean(log_prob_zk) - torch.mean(log_det)

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
