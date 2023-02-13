import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.glow import Glow, train_glow, generate_data
from src.utils.utils import display_images

# Load the MNIST dataset

mnist_train_loader, mnist_test_loader = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=512,
    batch_size=256,
    path_to_data="./src/data/"
)
n_rows = 28
n_cols = 28
n_channels = 1

# Create the model
model = Glow(
    width=10,
    depth=5,
    n_levels=3,
    input_dims=n_rows * n_cols * n_channels
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters())

# Train the model
n_epoch = 200
model = train_glow(model, optimizer, mnist_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)
