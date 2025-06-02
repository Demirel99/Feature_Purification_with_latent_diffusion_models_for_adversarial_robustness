import torch
import os

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Paths ---
# VAE model path (ensure this is where your trained VAE is saved)
VAE_MODEL_PATH = "vae_mnist_pytorch.pth"

# Directory for results, latents, and diffusion model
RESULTS_DIR_ROOT = "results_diffusion"
LATENT_DATA_DIR = os.path.join(RESULTS_DIR_ROOT, "latents")
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR_ROOT, "saved_models")
FIGURES_SAVE_DIR = os.path.join(RESULTS_DIR_ROOT, "figures")

# Ensure directories exist
os.makedirs(LATENT_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)

LATENT_TRAIN_PATH = os.path.join(LATENT_DATA_DIR, "latents_train.pt")
LATENT_VAL_PATH = os.path.join(LATENT_DATA_DIR, "latents_val.pt")
DIFFUSION_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "diffusion_model_mnist.pth")

# --- VAE Configuration (matching your existing VAE) ---
# These are from your config_vae.py, needed for consistency if using VAE parts
# For latent generation, LATENT_DIM_CHANNELS from config_vae is num_channels for diffusion input
VAE_LATENT_DIM_CHANNELS = 2 # from your config_vae.LATENT_DIM_CHANNELS
VAE_LATENT_H = 4 # from your config_vae.LATENT_H
VAE_LATENT_W = 4 # from your config_vae.LATENT_W
VAE_FEATURES_ENC = [1, 8, 16, 32] # from your config_vae.FEATURES_ENC

# --- Diffusion Model Hyperparameters (from notebook) ---
# U-Net architecture
UNET_FEATURES = [32, 64]  # Channels at different U-Net depths
TIME_EMBEDDING_DIM = 128   # Dimension of the time embedding
NUM_CHANNELS_LATENT = VAE_LATENT_DIM_CHANNELS # Input channels to U-Net = latent channels from VAE

# Variance schedule
TIME_STEPS = 1000
SCHEDULE_TYPE = "linear"  # "linear" or "cosine"
# For linear schedule
LINEAR_SCHEDULE_START = 1e-4
LINEAR_SCHEDULE_END = 7e-3 # Notebook uses 2e-2 for 2000 steps, 7e-3 was mentioned for 1000
# For cosine schedule
COSINE_SCHEDULE_S = 0.008

# --- Training Hyperparameters for Diffusion Model ---
EPOCHS_DIFFUSION = 200  # As in notebook
BATCH_SIZE_DIFFUSION = 128
LEARNING_RATE_DIFFUSION = 1e-3
I_LOG_DIFFUSION = 10    # Log every I_LOG_DIFFUSION epochs

# --- Generation Hyperparameters ---
NUM_SAMPLES_TO_GENERATE = 64 # e.g., 8x8 grid
N_ROW_PLOT = 8