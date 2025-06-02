import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

def plot_images(images, title="Generated Images", save_path=None, n_row=8):
    """
    Plots a batch of images.
    images: Tensor of shape (N, C, H, W)
    """
    plt.figure(figsize=(10, 10 * images.shape[0]/(n_row*images.shape[2]/images.shape[3]) )) # Adjust figure size
    plt.axis("off")
    plt.title(title)
    
    # Make a grid and plot
    grid = vutils.make_grid(images.cpu(), nrow=n_row, padding=2, normalize=False)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_losses(train_losses, val_losses, train_recon_losses, train_kl_losses, save_path=None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, label='Avg Train Total Loss')
    plt.plot(epochs_range, val_losses, label='Avg Val Total Loss')
    plt.xlabel('Epochs (x I_LOG)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Total Losses')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_recon_losses, label='Avg Train Recon Loss')
    plt.plot(epochs_range, train_kl_losses, label='Avg Train KL Divergence (as in article)')
    plt.xlabel('Epochs (x I_LOG)')
    plt.ylabel('Loss Component')
    plt.legend()
    plt.title('Loss Components (Training)')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    plt.show()