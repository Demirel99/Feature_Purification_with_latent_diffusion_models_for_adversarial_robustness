import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=128, root='./data'):
    transform = transforms.Compose([
        transforms.Pad(2), # Pad 28x28 to 32x32 (2 pixels on each side)
        transforms.ToTensor(), # Converts to [0, 1] range and CxHxW
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, _ = get_mnist_dataloaders(batch_size=4)
    data, labels = next(iter(train_loader))
    print("Batch shape:", data.shape) # Should be [4, 1, 32, 32]
    print("Batch min/max:", data.min(), data.max())