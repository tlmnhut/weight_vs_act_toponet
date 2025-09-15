import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from utils import load_test_dataset
from weight_cifar import DeeperCNN_CIFAR10
from weight_mnist import MinimalCNN_GAP_DO


def test_model_accuracy(model_dir, data_loader):
    """
    Test the accuracy of all models in the specified directory.

    Args:
        model_dir (str): Path to the directory containing model files.
        data_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        list: A list of accuracies for each model.
    """
    accuracies = []

    # Iterate through all model files in the directory
    for model_path in tqdm(sorted(glob.glob(os.path.join(model_dir, "*.pth")))):
        # Initialize and load model
        if 'MNIST' in model_dir:
            model = MinimalCNN_GAP_DO()
        else:
            model = DeeperCNN_CIFAR10()
        model_weights = torch.load(model_path, weights_only=True)
        model.load_state_dict(model_weights)
        model.eval()  # Set model to evaluation mode

        correct = 0
        total = 0

        # Evaluate model accuracy
        with torch.no_grad():
            for images, labels in data_loader:
                outputs, _ = model(images)  # Get model predictions
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy for the current model
        accuracy = correct / total
        accuracies.append(accuracy)

    return accuracies


def evaluate_models(model_dir, noise_type, noise_level=None):
    """
    Load the test dataset and evaluate the accuracy of all models in the specified directory.

    Args:
        model_dir (str): Path to the directory containing model files.

    Returns:
        list: A list of accuracies for each model.
    """
    # Load the test dataset
    if 'MNIST' in model_dir:
        dataset_name = 'mnist'
    else:
        dataset_name = 'cifar'
    test_dataset = load_test_dataset(dataset_name)

    # Add white noise to the dataset
    if noise_level is not None:
        if noise_type == 'white':
            test_dataset = add_white_noise(dataset_name, noise_level)
        elif noise_type == 'pink':
            test_dataset = add_pink_noise(dataset_name, noise_level)
        elif noise_type == 'salt_pepper':
            test_dataset = add_salt_pepper_noise(dataset_name, noise_level)
        else:
            raise ValueError("Unsupported noise type.")
        
    # Test model accuracy
    accuracies = test_model_accuracy(model_dir, test_dataset)

    return accuracies


def add_white_noise(dataset_name, noise_level=0.01):
    """
    Add white noise to the dataset and apply transformations back.

    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'cifar').
        noise_level (float): Standard deviation of the white noise.

    Returns:
        DataLoader: DataLoader for the dataset with added noise and transformations.
    """
    # Step 1: Load the dataset without normalization
    data_loader = load_test_dataset(dataset_name, use_norm=False)
    # data_loader = load_test_dataset(dataset_name, use_norm=True)

    # Step 2: Add white noise to the dataset
    noisy_images = []
    noisy_labels = []
    for images, labels in data_loader:
        # Add white noise to the images
        noisy_batch = images + noise_level * torch.randn_like(images)
        # Clip values to be in valid range [0, 1]
        # print(noisy_batch.min(), noisy_batch.max())
        noisy_batch = torch.clamp(noisy_batch, 0, 1)
        noisy_images.append(noisy_batch)
        noisy_labels.append(labels)

    # Concatenate all batches into a single tensor
    noisy_images = torch.cat(noisy_images, dim=0)
    noisy_labels = torch.cat(noisy_labels, dim=0)

    # Step 3: Add the normalization transformation back to the dataset
    if dataset_name == 'mnist':
        transform = transforms.Normalize((0.1307,), (0.3081,))  # Normalization values for MNIST
    elif dataset_name == 'cifar':
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalization values for CIFAR-10
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")

    # Apply normalization to the noisy images
    noisy_images = transform(noisy_images)

    # Create a TensorDataset and DataLoader
    noisy_dataset = TensorDataset(noisy_images, noisy_labels)
    noisy_data_loader = DataLoader(noisy_dataset, batch_size=256, shuffle=False)

    return noisy_data_loader


def add_pink_noise(dataset_name, noise_level=0.01):
    """
    Add pink noise to the dataset and apply transformations back.
    Pink noise has power spectral density inversely proportional to frequency.

    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'cifar').
        noise_level (float): Scale factor for the pink noise amplitude.

    Returns:
        DataLoader: DataLoader for the dataset with added pink noise and transformations.
    """
    # Step 1: Load the dataset without normalization
    data_loader = load_test_dataset(dataset_name, use_norm=False)
    # data_loader = load_test_dataset(dataset_name, use_norm=True)

    # Step 2: Add pink noise to the dataset
    noisy_images = []
    noisy_labels = []
    
    for images, labels in data_loader:
        # Get image dimensions
        batch_size, channels, height, width = images.shape
        
        # Generate pink noise for each image in the batch
        pink_noise_batch = []
        for _ in range(batch_size):
            # Generate pink noise for each channel
            channel_noise = []
            for _ in range(channels):
                # Generate white noise
                white = np.random.normal(0, 1, (height, width))
                
                # Convert to frequency domain
                f_white = np.fft.fft2(white)
                
                # Create pink noise filter (1/f spectrum)
                f_x = np.fft.fftfreq(width)
                f_y = np.fft.fftfreq(height)
                f_XX, f_YY = np.meshgrid(f_x, f_y)
                f_dist = np.sqrt(f_XX**2 + f_YY**2)
                f_dist[0, 0] = 1.0  # Avoid division by zero
                
                # Apply pink noise filter
                # pink_f = f_white / np.sqrt(f_dist)
                pink_f = f_white / f_dist
                
                # Convert back to spatial domain
                pink = np.real(np.fft.ifft2(pink_f))
                
                # Normalize
                pink = pink - pink.mean()
                pink = pink / pink.std()
                
                channel_noise.append(pink)
            
            # Stack channels
            pink_noise = np.stack(channel_noise)
            pink_noise_batch.append(pink_noise)
        
        # Convert to tensor and add to images
        pink_noise_batch = torch.tensor(np.stack(pink_noise_batch), dtype=torch.float32)
        noisy_batch = images + noise_level * pink_noise_batch
        
        # Clip values to be in valid range [0, 1]
        noisy_batch = torch.clamp(noisy_batch, 0, 1)
        
        noisy_images.append(noisy_batch)
        noisy_labels.append(labels)

    # Concatenate all batches into a single tensor
    noisy_images = torch.cat(noisy_images, dim=0)
    noisy_labels = torch.cat(noisy_labels, dim=0)

    # Step 3: Add the normalization transformation back to the dataset
    if dataset_name == 'mnist':
        transform = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == 'cifar':
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                      (0.2023, 0.1994, 0.2010))
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")

    # Apply normalization to the noisy images
    noisy_images = transform(noisy_images)

    # Create a TensorDataset and DataLoader
    noisy_dataset = TensorDataset(noisy_images, noisy_labels)
    noisy_data_loader = DataLoader(noisy_dataset, batch_size=256, shuffle=False)

    return noisy_data_loader


def add_salt_pepper_noise(dataset_name, noise_level=0.01):
    # https://www.askpython.com/python/examples/adding-noise-images-opencv
    """
    Add salt and pepper noise to the dataset by randomly selecting pixels.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'cifar').
        noise_level (float): Ratio of pixels to be corrupted with noise.

    Returns:
        DataLoader: DataLoader for the dataset with added salt and pepper noise.
    """
    # Step 1: Load the dataset without normalization
    data_loader = load_test_dataset(dataset_name, use_norm=False)

    # Step 2: Add salt and pepper noise to the dataset
    noisy_images = []
    noisy_labels = []
    
    for images, labels in data_loader:
        batch_size, channels, height, width = images.shape
        noisy_batch = images.clone()
        
        # Calculate number of pixels to corrupt per image
        n_pixels = int(height * width * noise_level)
        
        for idx in range(batch_size):
            # For each image in the batch
            for _ in range(n_pixels):
                # Randomly select pixel location
                row = torch.randint(0, height, (1,))
                col = torch.randint(0, width, (1,))
                
                # Randomly choose salt or pepper
                if torch.rand(1) < 0.5:
                    # Add pepper (black)
                    noisy_batch[idx, :, row, col] = 0.0
                else:
                    # Add salt (white)
                    noisy_batch[idx, :, row, col] = 1.0
        
        noisy_images.append(noisy_batch)
        noisy_labels.append(labels)

    # Concatenate all batches into a single tensor
    noisy_images = torch.cat(noisy_images, dim=0)
    noisy_labels = torch.cat(noisy_labels, dim=0)

    # Step 3: Add the normalization transformation back to the dataset
    if dataset_name == 'mnist':
        transform = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == 'cifar':
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                      (0.2023, 0.1994, 0.2010))
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")

    # Apply normalization to the noisy images
    noisy_images = transform(noisy_images)

    # Create a TensorDataset and DataLoader
    noisy_dataset = TensorDataset(noisy_images, noisy_labels)
    noisy_data_loader = DataLoader(noisy_dataset, batch_size=256, shuffle=False)

    return noisy_data_loader


if __name__ == "__main__":
    # # Define the model directory
    # model_dir = '../caos/allMNIST_models'
    # # Evaluate models and get accuracies
    # accuracies = evaluate_models(model_dir)
    # # Print the accuracies
    # paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    # for i, accuracy in enumerate(accuracies):
    #     print(f"Model {paths[i]}: Accuracy = {accuracy:.4f}")
    # # Optionally, save accuracies to a file or visualize them
    # np.save('./results/acc/mnist.npy', accuracies)

    # # load acc results
    # acc_mnist = np.load('./results/acc/mnist.npy')
    # acc_cifar = np.load('./results/acc/cifar.npy')

    noise_level = [0.01, 0.05, 0.1, 0.15, 0.2]
    all_accuracies_mnist, all_accuracies_cifar = [], []
    for noise in noise_level:
        # Define the model directory for MNIST
        model_dir_mnist = '../caos/allMNIST_models'
        # Evaluate models and get accuracies for MNIST
        accuracies_mnist = evaluate_models(model_dir_mnist, "white", noise_level=noise)
        # Print the accuracies for MNIST
        paths_mnist = sorted(glob.glob(os.path.join(model_dir_mnist, "*.pth")))
        print(f"MNIST with noise level {noise}:")
        for i, accuracy in enumerate(accuracies_mnist):
            print(f"Model {paths_mnist[i]}: Accuracy = {accuracy:.4f}")
        # Save accuracies to a file
        all_accuracies_mnist.append(accuracies_mnist)
        np.save(f'./results/acc/mnist_noise_white.npy', all_accuracies_mnist)

        # Define the model directory for CIFAR
        model_dir_cifar = '../caos/cifar'
        # Evaluate models and get accuracies for CIFAR
        accuracies_cifar = evaluate_models(model_dir_cifar, "white", noise_level=noise)
        # Print the accuracies for CIFAR
        paths_cifar = sorted(glob.glob(os.path.join(model_dir_cifar, "*.pth")))
        print(f"CIFAR with noise level {noise}:")
        for i, accuracy in enumerate(accuracies_cifar):
            print(f"Model {paths_cifar[i]}: Accuracy = {accuracy:.4f}")
        # Save accuracies to a file
        all_accuracies_cifar.append(accuracies_cifar)
        np.save(f'./results/acc/cifar_noise_white.npy', all_accuracies_cifar)


    # # load noise results
    # mnist_noise_results = np.load('./results/acc/mnist_noise_0.1.npy').reshape(13, -1)
    # cifar_noise_results = np.load('./results/acc/cifar_noise_0.1.npy').reshape(13, -1)
    # print(mnist_noise_results.mean(axis=1), mnist_noise_results.std(axis=1))
    # print(cifar_noise_results.mean(axis=1), cifar_noise_results.std(axis=1))
