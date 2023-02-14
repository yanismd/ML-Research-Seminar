import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.vae.vae import VAE, train_vae_inverse_lostdata, restore_lostdata_data
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

# Set the noise parameters
square_size = 4

# Create the model
model = VAE(
    h_dim_1=512,
    h_dim_2=256,
    z_dim=20,
    n_rows=n_rows,
    n_cols=n_cols,
    n_channels=n_channels
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters())

# Train the model
n_epoch = 500
model = train_vae_inverse_lostdata(
    model,
    optimizer,
    mnist_train_loader,
    n_epoch=n_epoch,
    square_size=square_size
)

# Try restoring data from the test set with the same noise applied as for the training set
target_data_list, noisy_data_list, restored_data_list = restore_lostdata_data(
    model,
    mnist_test_loader,
    square_size=square_size
)

# Display the results
display_restoration_process(
    target_data_list,
    noisy_data_list,
    restored_data_list,
    max_samples=5
)
