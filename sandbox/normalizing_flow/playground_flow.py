import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.classic.flow import FCNEncoder, FCNDecoder, FlowModel, train_flow, generate_data
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
z_dim = 40

# Create the model

encoder = FCNEncoder(
    hidden_sizes=[512, 256, 2*z_dim],
    dim_input=n_channels * n_cols * n_rows
)
flow_model = FlowModel(
    flows=[
              'PlanarFlow',
              'RadialFlow',
              'PlanarFlow',
          ] + ['PlanarFlow'] * 8,
    D=z_dim
)
decoder = FCNDecoder(
    hidden_sizes=[2*z_dim, 256, 512, n_channels * n_cols * n_rows],
    dim_input=z_dim
)

# Define the optimizer of the model
optimizer = optim.Adam(
    list(encoder.parameters()) + list(flow_model.parameters()) + list(decoder.parameters()),
    lr=10e-4
)

# Train the model
n_epoch = 100
model = train_flow(
    encoder, decoder, flow_model,
    optimizer,
    mnist_train_loader,
    n_epoch=n_epoch
)

# Generate new samples
generated_imgs = generate_data(flow_model, decoder, n_data=5, z_dim=z_dim)
# Display the results
display_images(generated_imgs)
