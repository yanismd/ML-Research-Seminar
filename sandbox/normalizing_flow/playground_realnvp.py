import torch.optim as optim
import torch

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.realnvp.realnvp import RealNVP, train_realnvp, generate_data
from src.utils.viz import display_images

# Load the MNIST dataset

mnist_train_loader, mnist_test_loader, (n_channels, n_rows, n_cols) = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=512,
    batch_size=256,
    path_to_data="../../src/data/"
)

# Create the model
num_layers = 5
masks = torch.nn.functional.one_hot(torch.tensor([i % n_cols for i in range(num_layers)])).float()

print(masks)

model = RealNVP(
    hidden_size=32,
    masks=masks
)

# Define the optimizer of the model
optimizer = optim.Adamax(model.parameters(), lr=0.1, weight_decay=5e-5)

# Train the model
n_epoch = 200
model = train_realnvp(model, optimizer, mnist_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)
