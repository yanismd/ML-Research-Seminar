import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.vae.vae import VAE, train_vae, generate_data
from src.utils.viz import display_images

# Load the MNIST dataset

mnist_train_loader, mnist_test_loader = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=512,
    batch_size=256,
    path_to_data="../../src/data/"
)
n_rows = 28
n_cols = 28
n_channels = 1

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
n_epoch = 200
model = train_vae(model, optimizer, mnist_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)
