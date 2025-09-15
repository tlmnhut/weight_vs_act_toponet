import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from datetime import datetime

from utils import create_data_loader, spatial_loss
from arch import DeeperCNN_CIFAR10
    

# Function to create a DataLoader in each process
# CAN increase num_workers to speed up based on resources.


# architecture different
def train_single_model(dataset, batch_size, model_id, metric, lambda_reg, epochs, model_save_path):
    # Initialize DataLoader inside the function to avoid sharing issues
    train_loader = create_data_loader(dataset, batch_size)
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print("created train loader")
    
    # We don't check for device; works faster in parallel CPU than bottleneck GPU
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Create a new model instance and move it to the GPU
    model = DeeperCNN_CIFAR10().to(device)   # Assume model is defined elsewhere
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    print(f"Model {model_id} - train_loader type: {type(train_loader)}")
    print(f"Model {model_id} - number of batches in train_loader: {len(train_loader)}")

    # Check the structure of the first batch in the train_loader
    first_batch = next(iter(train_loader), None)
    if first_batch:
        data, target = first_batch
        print(f"Model {model_id} - First batch data shape: {data.shape}")
        print(f"Model {model_id} - First batch target shape: {target.shape}")
    else:
        print(f"Model {model_id} - train_loader is empty.")
    
    for epoch in range(epochs):
        print("going through epochs")
        total_loss = 0
        print(f"Starting Epoch {epoch + 1} for Model {model_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for data, target in train_loader:
            # Move data and target to the GPU
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, fc1_activations = model(data)

            # Calculate cross-entropy loss
            ce_loss = F.cross_entropy(output, target)
            
             # Calculate spatial loss (commented out here, but add back if needed)
            weights = model.fc1.weight  # Get weights of the last fully connected layer
            sp_loss = spatial_loss(weights, metric)
            
            # Total loss
            loss = ce_loss + lambda_reg * sp_loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        
        print(f'Model {model_id}, Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
        print(f'Ended epoch {epoch + 1} for model {model_id}')

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    # Define transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 normalization values
    ])

    # Download and load the full CIFAR-10 dataset
    full_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    # Define the sizes for train, validation, and test sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Test dataset (CIFAR-10 test set)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    # Create DataLoader for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = DeeperCNN_CIFAR10()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # This will start the parallel training for each lambda value
    # lambda_values = [0.5, 1, 2, 3]  # Different values of lambda to test
    # lambda_values = [0.1, 0.3, 0.5, 1, 2, 3]  # Different values of lambda to test
    lambda_values = [1.0]  # Different values of lambda to test

    for lambda_reg in lambda_values:
        print(f"\nStarting training with lambda_reg = {lambda_reg}")
        
        # Get the start time
        start_time = datetime.now()
        print("Start Time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Call the parallel training function with current lambda value
        model_id = 0
        train_single_model(train_dataset, batch_size=64, model_id=model_id, metric='cosine', lambda_reg=lambda_reg, epochs=30,
                           model_save_path = f"test_cifar_weight_cosine_lambda{lambda_reg}_model{model_id}.pth")
        
        # Get the end time
        end_time = datetime.now()
        print("End Time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Training completed for lambda_reg = {lambda_reg}\n")