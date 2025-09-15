import torch
import torch.nn as nn
import torch.nn.functional as F


# dropout version. FC1 is computed separately from relu so can be captured with hook
class MinimalCNN_GAP_DO(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MinimalCNN_GAP_DO, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer (121 units)
        self.fc1 = nn.Linear(64, 121)  # Input size is 64 (from GAP), output size is 121
        # Dropout layer to prevent over-reliance on specific units
        self.dropout = nn.Dropout(p=dropout_rate)
        # Output layer
        self.fc2 = nn.Linear(121, 10)  # 10 classes for MNIST

    def forward(self, x):
        # Convolution + ReLU + Max Pooling Layer 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Convolution + ReLU + Max Pooling Layer 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, 1)  # Output shape will be (batch_size, 64, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)

        # Apply fc1 layer and capture activations before ReLU
        fc1_output = self.fc1(x)  # Capture pre-ReLU activations

        # Apply ReLU separately
        x = F.relu(fc1_output)

        # Apply dropout
        x = self.dropout(x)

        # Output layer (classification layer)
        x = self.fc2(x)

        return x, fc1_output  # Returning classification output only, not activations directly
    

class DeeperCNN_CIFAR10(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DeeperCNN_CIFAR10, self).__init__()
        
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 121)
        self.fc2 = nn.Linear(121, 10)  # Output for 10 classes (CIFAR-10)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers

        # Fully connected layers
        fc1_pre_relu = self.fc1(x)  # Compute fc1 output before ReLU
        fc1_post_relu = F.relu(fc1_pre_relu)  # Apply ReLU to fc1 output
        x = self.dropout(fc1_post_relu)
        x = self.fc2(x)

        return x, fc1_pre_relu
    
