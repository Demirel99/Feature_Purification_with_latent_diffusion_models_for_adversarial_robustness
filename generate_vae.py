import torch
import os

from config_vae import (DEVICE, FEATURES_ENC, LATENT_DIM_CHANNELS, LATENT_H, LATENT_W,
                        MODEL_SAVE_PATH, RESULTS_DIR)
from model_vae import VAE # Encoder, Decoder not needed if VAE encapsulates
from plot_utils import plot_images
import torchvision.transforms.functional as TF # For potential cropping if needed for viz

def generate_samples(num_samples=32):
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model weights not found at {MODEL_SAVE_PATH}. Train the model first.")
        return

    print(f"Using device: {DEVICE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = VAE(
        encoder_base_features=FEATURES_ENC,
        latent_dim_channels=LATENT_DIM_CHANNELS
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    with torch.no_grad():
        # Sample z from standard normal N(0,I)
        # Latent space is (LATENT_DIM_CHANNELS, LATENT_H, LATENT_W)
        z_sample = torch.randn(num_samples, LATENT_DIM_CHANNELS, LATENT_H, LATENT_W).to(DEVICE)
        
        # Use only the decoder part of the VAE model
        generated_images_32x32 = model.decoder_model(z_sample)
        generated_images_clipped = torch.clip(generated_images_32x32, 0.0, 1.0)

    # Output is 32x32. Visualize as is, or crop if desired for 28x28 view.
    # For consistency with notebook (which pads then likely visualizes padded):
    plot_images(generated_images_clipped, title=f"Generated VAE Samples ({num_samples}, 32x32)",
                save_path=os.path.join(RESULTS_DIR, "vae_generated_samples.png"),
                n_row=8 if num_samples >=8 else int(num_samples**0.5))

if __name__ == '__main__':
    generate_samples(num_samples=32) # Notebook generates 4*8=32 samples