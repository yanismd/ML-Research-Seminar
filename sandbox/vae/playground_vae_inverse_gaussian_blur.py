import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.vae.vae import VAE, train_vae_inverse_blur, restore_blur_data
from src.utils.viz import display_restoration_process

# Load the MNIST dataset

mnist_train_loader, mnist_test_loader = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=1000,
    batch_size=256,
    path_to_data="../../src/data/"
)
n_rows = 28
n_cols = 28
n_channels = 1

# Set the blur parameters
kernel_size = 7
sigma = 5

# Create the model
model = VAE(
    hidden_sizes_encoder=[512, 256],
    hidden_sizes_decoder=[256, 512],
    z_dim=20,
    n_rows=n_rows,
    n_cols=n_cols,
    n_channels=n_channels
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters(), lr=10e-4)

# Train the model
n_epoch = 500
model = train_vae_inverse_blur(
    model,
    optimizer,
    mnist_train_loader,
    n_epoch=n_epoch,
    kernel_size=kernel_size,
    sigma=sigma
)

# Try restoring data from the test set with the same noise applied as for the training set
target_data_list, noisy_data_list, restored_data_list = restore_blur_data(
    model,
    mnist_test_loader,
    kernel_size=kernel_size,
    sigma=sigma
)

# Display the results
display_restoration_process(
    target_data_list,
    noisy_data_list,
    restored_data_list,
    max_samples=5
)
