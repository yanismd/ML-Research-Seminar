from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.glow import Glow

# Load the dataset
mnist_train_loader, mnist_test_loader = fetch_mnist_loader(path_to_data="./src/data/")

# Define the input size, which corresponds to a flattened picture of the MNIST dataset
input_size = 28 * 28 * 1

model = Glow(width=10, depth=5, n_levels=10, input_dims=input_size)
