import torch
import os
from tqdm import tqdm

# User's existing modules
from model_vae import VAE # Assuming VAE class is defined here and matches training
from data_utils import get_mnist_dataloaders # For [0,1] normalized & padded data

# Diffusion config for paths and VAE params
import config_diffusion as cd
import config_vae # To get VAE hyperparams used during its training

def generate_and_save_latents():
    print(f"Using device: {cd.DEVICE}")

    # --- Load VAE Model ---
    print(f"Loading VAE model from {cd.VAE_MODEL_PATH}...")
    # Ensure VAE hyperparams match the VAE that was trained
    vae_model = VAE(
        encoder_base_features=config_vae.FEATURES_ENC, # From original VAE config
        latent_dim_channels=config_vae.LATENT_DIM_CHANNELS  # From original VAE config
    ).to(cd.DEVICE)
    
    try:
        vae_model.load_state_dict(torch.load(cd.VAE_MODEL_PATH, map_location=cd.DEVICE))
    except FileNotFoundError:
        print(f"ERROR: VAE model file not found at {cd.VAE_MODEL_PATH}. Please train the VAE first.")
        return
    except Exception as e:
        print(f"ERROR: Could not load VAE model: {e}")
        return
        
    vae_model.eval()
    print("VAE model loaded successfully.")

    # --- Get MNIST DataLoaders ---
    # Using batch size from diffusion config, but can be different
    train_loader, test_loader = get_mnist_dataloaders(batch_size=cd.BATCH_SIZE_DIFFUSION)
    print("MNIST dataloaders loaded.")

    # --- Generate Latents ---
    def get_latents_from_loader(loader, vae_model_instance):
        all_latents = []
        with torch.no_grad():
            for (data, _) in tqdm(loader, desc="Encoding data to latents"):
                data = data.to(cd.DEVICE) # Data is [B, 1, 32, 32], range [0,1]
                
                # Get mu and sigma_params from VAE encoder
                mu_sigma_map = vae_model_instance.encoder_model(data)
                mu = mu_sigma_map[:, :vae_model_instance.latent_dim_channels, :, :]
                sigma_params = mu_sigma_map[:, vae_model_instance.latent_dim_channels:, :, :]
                
                # Reparameterize to get z (latents) using VAE's reparameterization
                # This ensures latents are sampled correctly based on how VAE was trained
                z = vae_model_instance.reparameterize(mu, sigma_params)
                all_latents.append(z.cpu())
        return torch.cat(all_latents, dim=0)

    print("Generating training latents...")
    latents_train = get_latents_from_loader(train_loader, vae_model)
    torch.save(latents_train, cd.LATENT_TRAIN_PATH)
    print(f"Training latents saved to {cd.LATENT_TRAIN_PATH}, shape: {latents_train.shape}")

    print("Generating validation latents...")
    latents_val = get_latents_from_loader(test_loader, vae_model)
    torch.save(latents_val, cd.LATENT_VAL_PATH)
    print(f"Validation latents saved to {cd.LATENT_VAL_PATH}, shape: {latents_val.shape}")

    print("Latent generation complete.")

if __name__ == '__main__':
    generate_and_save_latents()