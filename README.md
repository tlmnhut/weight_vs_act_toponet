# Robustness and representational reorganization in topographic CNNs

We implemented topographic convolutional neural networks with two spatial constraints: Weight Similarity and Activation Similarity.

We evaluate the resulting models on classification accuracy, robustness to weight perturbations and input perturbations, and the spatial organization of learned representations.

## Code structure

- **arch.py** 
  - Defines CNN architectures for MNIST and CIFAR-10

- **train_mnist.py**
  - Training pipeline for MNIST dataset

- **train_cifar.py**
  - Training pipeline for CIFAR-10

- **test_acc_topo.py**
  - Evaluates model accuracy
  - Analyzes topological properties
  - Computes spatial statistics (Moran's I)

- **test_noise_images.py**
  - Tests model robustness against various types of noise

- **plot_fig.py**
  - Plot figures

- **utils.py**
  - Spatial loss functions

## Acknowledgment

Part of the code was developed by Uri Hasson.
