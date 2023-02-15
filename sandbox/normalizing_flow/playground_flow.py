import torch
import torch.nn.functional as F
import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.classic.flow import FlowModel, train_flow, generate_data
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

# Define the encoding dimension
z_dim = 20

# Create the model

model = FlowModel(
    flows=['PlanarFlow'] * 2,
    hidden_sizes_encoder=[512, 256],
    hidden_sizes_decoder=[256, 512],
    z_dim=z_dim,
    n_channels=n_channels,
    n_rows=n_rows,
    n_cols=n_cols,
    activation=F.elu
)

# Define the optimizer of the model
optimizer = optim.Adam(
    model.parameters(),
    lr=10e-5
)

# Train the model
n_epoch = 30
model = train_flow(
    model,
    optimizer,
    mnist_train_loader,
    n_epoch=n_epoch
)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)
