import torch

# Training Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200 # As per notebook's VAE section
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
I_LOG = 10 # Log every I_LOG epochs

# Model Hyperparameters for VAE (from notebook)
# features defines the number of channels at each encoder stage BEFORE the latent layer projection
FEATURES_ENC = [1, 8, 16, 32] # Input channels, then channels after each ConvEncoder block part
LATENT_DIM_CHANNELS = 2      # Number of channels for mu map and sigma map respectively
                             # Encoder will output 2 * LATENT_DIM_CHANNELS

# Latent map spatial dimensions derived from 32x32 input and len(FEATURES_ENC)-1 maxpools
# 32 // (2^(4-1)) = 32 // 8 = 4
LATENT_H = 4
LATENT_W = 4

# Paths
MODEL_SAVE_PATH = "vae_mnist_pytorch.pth" # Single file for the VAE model
RESULTS_DIR = "results_vae"