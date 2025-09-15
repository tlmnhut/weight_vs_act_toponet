import torch
import torch.nn.functional as F
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import libpysal.weights as ps
# from esda.moran import Moran
from torch.multiprocessing import Pool
from functools import partial
from scipy.signal import convolve2d
import pandas as pd

from utils import load_test_dataset
from weight_cifar import DeeperCNN_CIFAR10
from weight_mnist import MinimalCNN_GAP_DO



def load_model(model_path):
    """
    Load a PyTorch model from a .pth file.
    """
    # init the model architecture, then load the state dict

    if "cifar" in model_path:
        model = DeeperCNN_CIFAR10()
    else:
        model = MinimalCNN_GAP_DO()
    # Load the model state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # Set the model to evaluation mode
    model.eval()
    return model


def test_model_accuracy(model, test_loader, device):
    """
    Test the accuracy of the model on the test set.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            # If outputs is a tuple, use the first element (logits)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def load_and_test_acc(models_dir):
    # Iterate through all .pth files in the models directory
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pth"):
            model_path = os.path.join(models_dir, model_file)
            model = load_model(model_path).to(device)
            if "cifar" in model_file:
                test_loader = load_test_dataset("cifar")
            else:
                test_loader = load_test_dataset("mnist")
            # Test the model accuracy
            accuracy = test_model_accuracy(model, test_loader, device)
            print(f"Model: {model_file}, Accuracy: {accuracy:.2%}")


def visualize_fc1_pre_relu(models_dir, dataset_name):
    """
    Visualize 20 random images and their corresponding 11x11 grids from fc1_pre_relu values.
    """
    # Load the test set based on the dataset name
    test_loader = load_test_dataset(dataset_name)

    # Load all models with the dataset name in their file name
    model_files = [f for f in os.listdir(models_dir) if dataset_name in f and f.endswith(".pth")]
    models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if dataset_name == "cifar":
            model = DeeperCNN_CIFAR10()
        else:
            model = MinimalCNN_GAP_DO()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model.to(device))

    # Take 20 random images from the test set
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    indices = random.sample(range(images.size(0)), 20)
    selected_images = images[indices].to(device)

    # Prepare visualization
    fig, axes = plt.subplots(len(models) + 1, 20, figsize=(20, 10))  # Adjust grid size for 20 images
    
    # Visualize the images in the top row
    for i, img in enumerate(selected_images):
        img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        if dataset_name == "cifar":
            img = (img * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])  # Denormalize CIFAR
        else:
            img = img.squeeze()  # For MNIST, remove channel dimension
        img = np.clip(img, 0, 1)  # Clip values to [0, 1]
        axes[0, i].imshow(img, cmap="gray" if dataset_name == "mnist" else None)
        axes[0, i].axis('off')

    # Visualize the 11x11 grids from fc1_pre_relu values for each model
    for model_idx, model in enumerate(models):
        with torch.no_grad():
            fc1_pre_relu_values = model(selected_images)[1]  # Get fc1_pre_relu values
            for img_idx, fc1_pre_relu in enumerate(fc1_pre_relu_values):
                grid = fc1_pre_relu.view(11, 11).cpu().numpy()  # Reshape to 11x11
                axes[model_idx + 1, img_idx].imshow(grid, cmap='viridis')
                axes[model_idx + 1, img_idx].axis('off')

    # Add titles
    axes[0, 0].set_title("Images")
    for model_idx, model_file in enumerate(model_files):
        axes[model_idx + 1, 0].set_title(f"Model: {model_file}")

    plt.tight_layout()
    plt.show()


def plot_weights_histogram_by_layer(models_dir, dataset_name):
    """
    Compare weights for each layer across all models with either 'mnist' or 'cifar' in their name.
    The plot will have a grid of 3 x n (3 models and n layers).
    """
    # Load all models with "mnist" or "cifar" in their name
    model_files = [f for f in os.listdir(models_dir) if (dataset_name in f) and f.endswith(".pth")]
    models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if "cifar" in model_file:
            model = DeeperCNN_CIFAR10()
        else:
            model = MinimalCNN_GAP_DO()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append((model.to(device), model_file))

    # Collect weights for each layer across all models
    layer_names = set()  # To store unique layer names
    model_layer_weights = []  # List to store weights for each model and layer
    for model, model_file in models:
        model_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:  # Only consider weights (not biases)
                model_weights[name] = param.cpu().detach().numpy().flatten()
                layer_names.add(name)
        model_layer_weights.append((model_file, model_weights))

    # Sort layer names to ensure consistent order
    layer_names = sorted(layer_names)

    # Create a grid of subplots (3 models x n layers)
    fig, axes = plt.subplots(len(models), len(layer_names), figsize=(5 * len(layer_names), 5 * len(models)))
    for layer_idx, layer_name in enumerate(layer_names):
        # Dynamically calculate bins for this layer across all models
        all_layer_weights = []
        for _, model_weights in model_layer_weights:
            if layer_name in model_weights:
                all_layer_weights.extend(model_weights[layer_name])
        min_val, max_val = min(all_layer_weights), max(all_layer_weights)
        bins = np.linspace(min_val, max_val, 50)  # Dynamic bins based on min and max values

        for model_idx, (model_file, model_weights) in enumerate(model_layer_weights):
            ax = axes[model_idx, layer_idx]
            if layer_name in model_weights:
                ax.hist(model_weights[layer_name], bins=bins, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f"{model_file}\nLayer: {layer_name}")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            ax.grid(True)

    plt.tight_layout()
    plt.show()


# def compute_morans_i(matrix, spatial_weights):
#     """
#     Computes Moran's I for a given 2D numpy matrix with Moore neighborhood.

#     Args:
#         matrix (np.ndarray): An 11x11 numpy matrix.

#     Returns:
#         float: The calculated Moran's I value.
#     """
#     # Calculate Moran's I using PySAL's Moran class
#     # PySAL handles the mean centering and all the summation details.
#     moran = Moran(matrix.flatten(), spatial_weights)
#     return moran.I


def batch_morans_I(images):
    """
    images: numpy array of shape [B, 11, 11]
    returns: numpy array of shape [B] with Global Moran's I per image
    """
    B, H, W = images.shape
    N = H * W

    # Moore kernel: 3x3 with 1s excluding center
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Precompute W (sum of all weights per image)
    # Same for all images if no mask
    weight_mask = convolve2d(np.ones((H, W)), kernel, mode='same', boundary='fill', fillvalue=0)
    # weight_mask = convolve2d(np.ones((H, W)), kernel, mode='same', boundary='wrap')
    W = weight_mask.sum()

    # Mean center each image
    x_bar = images.mean(axis=(1, 2), keepdims=True)
    x_centered = images - x_bar  # shape: [B, 11, 11]

    x_centered_torch = torch.tensor(x_centered, dtype=torch.float32).unsqueeze(1)  # [B, 1, 11, 11]
    kernel_torch = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    neighbor_sum = F.conv2d(x_centered_torch, kernel_torch, padding=1)  # [B, 1, 11, 11]
    neighbor_sum = neighbor_sum.squeeze(1).numpy()  # back to numpy, shape [B, 11, 11]

    # Numerator: sum over x_i' * sum_j(x_j')
    numerator = np.sum(x_centered * neighbor_sum, axis=(1, 2))  # shape: [B]

    # Denominator: sum over x_i'^2
    denominator = np.sum(x_centered ** 2, axis=(1, 2))  # shape: [B]

    # Moran's I
    I = (N / W) * (numerator / denominator)
    return I


def analyze_spatial_correlation(models_dir, dataset_name, save_path):
    """
    Analyze spatial correlation in fc1_pre_relu activations using Moran's I with parallel processing.
    
    Args:
        models_dir (str): Directory containing model files
        dataset_name (str): Name of dataset ('mnist' or 'cifar')
        save_path (str): Path to save results
    """
    # Load test dataset
    test_loader = load_test_dataset(dataset_name)
    
    # Load models
    model_files = sorted([f for f in os.listdir(models_dir)])
    results = {}
    
    # Create a partial function with fixed spatial_weights
    spatial_weights = ps.lat2W(11, 11, rook=False)  # Moore neighborhood for 11x11 grid
    # compute_morans_i_partial = partial(compute_morans_i, spatial_weights=spatial_weights)
    
    for model_file in tqdm(model_files):
        # print(f"Processing {model_file}")
        model_path = os.path.join(models_dir, model_file)
        
        # Load model
        if dataset_name == "cifar":
            model = DeeperCNN_CIFAR10()
        else:
            model = MinimalCNN_GAP_DO()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.to(device)
        
        # Process batch of images
        morans_i_scores = []
        batch_activations = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                _, fc1_pre_relu = model(data)
                batch_activations.append(fc1_pre_relu.cpu().numpy())  # Store activations
                # # Prepare grids for parallel processing
                # grids = [activation.view(11, 11).cpu().numpy() 
                #         for activation in fc1_pre_relu]
                
                # # Process batch in parallel
                # with Pool() as pool:
                #     batch_scores = pool.map(compute_morans_i_partial, grids)
                # morans_i_scores.extend(batch_scores)
        
        batch_activations = np.concatenate(batch_activations, axis=0).reshape(-1, 11, 11)
        # Compute Moran's I for the entire batch of activations
        morans_i_scores = batch_morans_I(batch_activations)
    
        # Store results
        results[model_file] = morans_i_scores
    
    # Save results
    np.save(save_path, results, allow_pickle=True)


def process_morans_i_results(file_path):
    results = np.load(file_path, allow_pickle=True).item()
    # stack all results into a single array
    all_scores = []
    for model_file, scores in results.items():
        all_scores.append(scores)
    all_scores = np.array(all_scores) # shape 130x10000
    # group each 10 consecutive rows together
    all_scores = all_scores.reshape(-1, 10, all_scores.shape[1])  # shape (13, 10, 10000)
    # all_scores = all_scores.reshape(all_scores.shape[0], -1)  # shape (13, 10*10000)
    # compute mean and std and sem
    # mean_scores = np.mean(all_scores, axis=2)
    # std_scores = np.std(all_scores, axis=1)
    # sem_scores = std_scores / np.sqrt(all_scores.shape[1])  # standard error of the mean
    mean_scores = all_scores.mean(axis=2).mean(axis=1)  # shape (13,)
    std_scores = all_scores.mean(axis=2).std(axis=1)  # shape (13,)
    # print(mean_scores.shape, std_scores.shape)
    sem_scores = std_scores / np.sqrt(10)  # standard error of the mean
    # print results
    labels = ['ctrl', 'as_0.1', 'as_0.3', 'as_0.5', 'as_1', 'as_2', 'as_3',
                      'ws_0.1', 'ws_0.3', 'ws_0.5', 'ws_1', 'ws_2', 'ws_3']
    # labels = ['as_0.1', 'as_0.3', 'as_0.5', 'as_1', 'as_2', 'as_3']
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
        print(f"{labels[i]}: {mean:.2f} ({std:.2f})")

    ctrl_condition = mean_scores[0]
    as_conditions = mean_scores[1:7]
    ws_conditions = mean_scores[7:]
    # as_conditions = mean_scores[:]
    lambda_values = [0.1, 0.3, 0.5, 1, 2, 3]
    

    # create a summary DataFrame
    sem_summary = pd.DataFrame({
        'Condition': ['as'] * 6 + ['ws'] * 6,
        # 'Condition': ['as'] * 6,
        'Lambda': lambda_values * 2,
        # 'Lambda': lambda_values * 1,
        'moran_i_mean': np.concatenate([as_conditions, ws_conditions]),
        # 'moran_i_mean': np.concatenate([as_conditions]),
        'moran_i_sem': np.concatenate([sem_scores[1:7], sem_scores[7:]])
        # 'moran_i_sem': np.concatenate([sem_scores[:]])
    })
    control_summary = pd.DataFrame({
        'Condition': ['control'],
        'Lambda': [0],
        'moran_i_mean': [ctrl_condition],
        'moran_i_sem': [sem_scores[0]]
    })
    # control_summary = None
    return sem_summary, control_summary


def process_ed_results(dataset_name):
    ed_weight = np.load(f"./results/stat/ed_weight_{dataset_name}.npy", allow_pickle=True)
    ed_act = np.load(f"./results/stat/ed_act_{dataset_name}.npy", allow_pickle=True)
    grouped_ed_weight = np.reshape(ed_weight, (13, 10)) # 13 condtions, 10 models each
    grouped_ed_act = np.reshape(ed_act, (13, 10))

    # Calculate the average and SEM for each group of ten
    avg_weight = np.mean(grouped_ed_weight, axis=1)
    sem_weight = np.std(grouped_ed_weight, axis=1, ddof=1) / np.sqrt(grouped_ed_weight.shape[1])
    avg_act = np.mean(grouped_ed_act, axis=1)
    sem_act = np.std(grouped_ed_act, axis=1, ddof=1) / np.sqrt(grouped_ed_act.shape[1])

    return avg_weight, sem_weight, avg_act, sem_act


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_and_test_acc(models_dir="./result/model/")
    # visualize_fc1_pre_relu(models_dir="./result/model/", dataset_name="cifar")
    # plot_weights_histogram_by_layer(models_dir="./result/model/", dataset_name="cifar")
    
    # analyze_spatial_correlation(
    #     models_dir="../caos/actsmoothGlobalmodels/",
    #     dataset_name="mnist",
    #     save_path="./results/stat/mnist_morans_i_batch_fill_as_global.npy"
    # )
    process_morans_i_results(file_path="./results/stat/mnist_morans_i_batch_fill_as_global.npy")
    
    # import numpy as np
    # morans_i = np.load("./results/stat/cifar_morans_i.npy", allow_pickle=True).item()
    # morans_i_batch_fill = np.load("./results/stat/cifar_morans_i_batch_fill.npy", allow_pickle=True).item()
    # morans_i_batch_wrap = np.load("./results/stat/cifar_morans_i_batch_wrap.npy", allow_pickle=True).item()
    # for model_file, scores in morans_i.items():
    #     print(f"{model_file}: {np.mean(scores):.4f} ({np.std(scores):.4f})")
    # for model_file, scores in morans_i_batch_fill.items():
    #     print(f"{model_file}: {np.mean(scores):.4f} ({np.std(scores):.4f})")
    # for model_file, scores in morans_i_batch_wrap.items():
    #     print(f"{model_file}: {np.mean(scores):.4f} ({np.std(scores):.4f})")
