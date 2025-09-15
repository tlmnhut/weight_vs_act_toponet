import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def create_data_loader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)


def spatial_loss(weights, metric='euclidean'):
    # Convert weights to an 11x11 grid
    grid_size = 11
    # weights = weights.view(grid_size, grid_size, 256)
    weights = weights.view(grid_size, grid_size, -1)

    # Initialize loss
    loss = 0.0
    num_neighbors = 0

    # Define the neighbor offsets (8 possible neighbors)
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        (0, -1),          (0, 1),     # Left,        , Right
        (1, -1), (1, 0), (1, 1)       # Bottom-left, Bottom, Bottom-right
    ]

    # Iterate over each unit in the 11x11 grid
    for i in range(grid_size):
        for j in range(grid_size):
            current_weight = weights[i, j]

            # Initialize a list to hold the neighbors
            neighbors = []

            # Find valid neighbors based on the current position
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:  # Check bounds
                    neighbors.append(weights[ni, nj])

            # Compute pairwise distances between the current weight and its neighbors
            for neighbor in neighbors:
                if metric == 'euclidean':
                    distance = torch.norm(current_weight - neighbor)
                elif metric == 'cosine':
                    distance = cosine_distance(current_weight, neighbor)
                elif metric == 'correlation':
                    distance = correlation_distance(current_weight, neighbor)
                else:
                    distance = 0
                # # Compute the squared distance (Euclidean)
                # distance = torch.norm(current_weight - neighbor)
                loss += distance

            # Count the number of neighbors considered
            num_neighbors += len(neighbors)

    # Average loss over the number of neighbor comparisons
    return loss / num_neighbors if num_neighbors > 0 else loss


def correlation_distance(x, y):
    # Subtract means
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    # Compute numerator (covariance) and denominator (product of std deviations)
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum()) * torch.sqrt((y_centered ** 2).sum())
    
    # Add a small epsilon to denominator for numerical stability
    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)
    
    # Compute correlation distance
    distance = 1 - correlation
    return distance


def cosine_distance(x, y):
    # Ensure tensors are of a floating-point type
    x = x.float()
    y = y.float()
    
    # Compute numerator (dot product)
    numerator = (x * y).sum()
    
    # Compute denominator (product of L2 norms)
    denominator = torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum())
    
    # Add a small epsilon for numerical stability
    epsilon = 1e-8
    similarity = numerator / (denominator + epsilon)
    
    # Compute cosine distance
    distance = 1 - similarity
    return distance


def load_test_dataset(dataset_name, use_norm=True):
    """
    Load the test dataset with optional normalization.

    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'cifar').
        use_norm (bool): Whether to apply normalization in the transforms.

    Returns:
        DataLoader: DataLoader for the specified test dataset.
    """
    # Define MNIST transform
    transform_mnist = [transforms.ToTensor()]
    if use_norm:
        transform_mnist.append(transforms.Normalize((0.1307,), (0.3081,)))  # Normalization values for MNIST

    # Define CIFAR transform
    transform_cifar = [transforms.ToTensor()]
    if use_norm:
        transform_cifar.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))  # Normalization values for CIFAR-10

    # Compose transforms
    transform_mnist = transforms.Compose(transform_mnist)
    transform_cifar = transforms.Compose(transform_cifar)

    # Load dataset
    if dataset_name == "mnist":
        dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_mnist)
    elif dataset_name == "cifar":
        dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_cifar)
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")

    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=10)


def analyze_intra_condition_similarity(sim_matrix):
    """
    Analyzes the similarity within each condition block (10x10)
    of a similarity matrix.

    Args:
        sim_matrix (np.ndarray): The square similarity matrix (e.g., 130x130).

    Returns:
        dict: A dictionary where keys are condition labels and values are
              tuples of (mean_similarity, std_similarity) for the upper
              triangle within that condition block.
    """
    # labels = ['ctrl',
    #           'as_0.1', 'as_0.3', 'as_0.5', 'as_1', 'as_2', 'as_3',
    #           'ws_0.1', 'ws_0.3', 'ws_0.5', 'ws_1', 'ws_2', 'ws_3']
    # analysis_results = {}
    analysis_results = []

    # for i, label in enumerate(labels):
    for i in range(13):
        start_idx = i * 10
        end_idx = start_idx + 10

        # Extract the 10x10 block for the current condition
        condition_block = sim_matrix[start_idx:end_idx, start_idx:end_idx]

        # Get the indices of the upper triangle (excluding the diagonal)
        upper_triangle_indices = np.triu_indices(10, k=1)

        # Extract values from the upper triangle
        upper_triangle_values = condition_block[upper_triangle_indices]

        # Compute mean and standard deviation
        mean_sim = np.mean(upper_triangle_values)
        std_sim = np.std(upper_triangle_values)
        se_sim = std_sim / np.sqrt(len(upper_triangle_values)) # standard error

        # analysis_results[label] = (mean_sim, se_sim)
        analysis_results.append([mean_sim, se_sim])

    return analysis_results
