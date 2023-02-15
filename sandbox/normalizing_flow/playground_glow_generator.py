import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.glow.glow import Glow, train_glow, generate_data
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
flow_steps = 2
L = 2  # Dimension reduction parameter (reduce by 2 the size of the data at each value)

model = Glow(
    image_shape=(n_channels, n_rows, n_cols),
    hidden_channels=30,
    K=flow_steps,
    L=L,
    actnorm_scale=100,
    flow_permutation="invconv",
    flow_coupling="affine",
    LU_decomposed=True,
    y_classes=None,
    learn_top=True,
    y_condition=False
)
model.set_actnorm_init()
model.eval()

# Define the optimizer of the model
optimizer = optim.Adamax(model.parameters(), lr=0.1, weight_decay=5e-5)

# Train the model
n_epoch = 200
model = train_glow(model, optimizer, mnist_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, temperature=0.7)
# Display the results
display_images(generated_imgs)
