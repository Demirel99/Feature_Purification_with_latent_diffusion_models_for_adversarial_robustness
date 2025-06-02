# dataset.py
import torch
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF

def get_processed_data(data_root='.'):
    """
    Downloads, loads, and preprocesses the MNIST dataset.
    Preprocessing includes:
    1. Normalizing pixel values to [0, 1).
    2. Padding images from 28x28 to 32x32.
    Returns:
        x_train (Tensor): Training images.
        y_train (Tensor): Training labels.
        x_val (Tensor): Validation (test) images.
        y_val (Tensor): Validation (test) labels.
    """
    print("Loading and preprocessing MNIST dataset...")
    # Download and load dataset
    data_train = MNIST(data_root, download=True, train=True)
    data_test = MNIST(data_root, download=True, train=False)

    # Normalize to [0, 1)
    # Ensure data is float before division
    x_train = data_train.data.float() / 255.0
    x_val = data_test.data.float() / 255.0

    y_train = data_train.targets
    y_val = data_test.targets

    # Pad images with a 2-pixel border to make them 32x32
    # Padding tuple is (pad_left, pad_right, pad_top, pad_bottom)
    # PyTorch's pad function expects (N, C, H, W) or (C, H, W) or (H,W)
    # Since our data is (N, H, W), we can pad directly on the last two dimensions
    x_train = torch.nn.functional.pad(x_train, (2, 2, 2, 2), value=0.0)
    x_val = torch.nn.functional.pad(x_val, (2, 2, 2, 2), value=0.0)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print("Dataset loaded and preprocessed.")
    return x_train, y_train, x_val, y_val

if __name__ == '__main__':
    # Example usage:
    x_train, y_train, x_val, y_val = get_processed_data()
    print("\n--- Sample Data Info ---")
    print(f"x_train shape: {x_train.shape}, dtype: {x_train.dtype}, min: {x_train.min()}, max: {x_train.max()}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"x_val shape: {x_val.shape}, dtype: {x_val.dtype}, min: {x_val.min()}, max: {x_val.max()}")
    print(f"y_val shape: {y_val.shape}, dtype: {y_val.dtype}")

    # Optional: visualize a sample
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(10,2))
    for i in range(5):
        axes[i].imshow(x_train[i], cmap='gray')
        axes[i].set_title(f"Label: {y_train[i].item()}")
        axes[i].axis('off')
    plt.show()