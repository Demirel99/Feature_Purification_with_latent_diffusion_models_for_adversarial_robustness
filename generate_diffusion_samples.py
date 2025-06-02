import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # <--- ADD THIS LINE

# Diffusion model components
from diffusion_model import TimeEmbeddingModel
from diffusion_utils import get_alpha_prod
import config_diffusion as cd

# VAE model (for decoder) and config
from model_vae import VAE # Or just ConvDecoder if you prefer to load only that part
import config_vae # For VAE structure if loading full VAE

# User's plot_utils
from plot_utils import plot_images


def denoise_at_t(model, x_t, t_tensor, alpha_prod_t, alpha_prod_t_prev, device):
    """
    Performs one step of the reverse diffusion process.
    model: The trained U-Net diffusion model.
    x_t: Current noised sample at step t. [B, C, H, W]
    t_tensor: Current time step t, as a tensor for the model. [B]
    alpha_prod_t: $\bar{\alpha}_t$
    alpha_prod_t_prev: $\bar{\alpha}_{t-1}$
    """
    if t_tensor[0].item() == 1: # If t=1, alpha_prod_t_prev is for t=0, effectively alpha_prod_0 = 1.0
                                # However, alpha_prod_t_prev here should be for alpha_prod[t-2] in array
                                # So if t=1 (idx 0), t-1 has no defined alpha_prod_t_prev in schedule array for t=0.
                                # The formula uses alpha_t not alpha_prod_t_prev.
                                # alpha_t = alpha_prod_t / alpha_prod_t_prev
        # For t=1, there is no "previous" alpha_prod in the schedule directly.
        # The term sigma_t * z involves (1-alpha_prod_t_prev)/(1-alpha_prod_t) * (1-alpha_t)
        # When t=1, noise is not added in DDPM sampling. For DDIM, it can be.
        # Notebook version:
        # a = alpha_prod[t-1] / alpha_prod[t-2] if t > 1 else alpha_prod[0]
        # Here, alpha_prod_t is alpha_prod[t-1], alpha_prod_t_prev is alpha_prod[t-2]
        pass # Noise is not added for t=1 in the notebook's DDPM-like sampling loop

    # Predict noise using U-Net
    eps_theta = model(x_t, t_tensor) # t_tensor should be (B,)

    # Calculate x_{t-1} mean component
    # Coeff for x_t: 1 / sqrt(alpha_t)
    # Coeff for eps_theta: (1 - alpha_t) / sqrt(1 - alpha_bar_t)
    # where alpha_t = alpha_prod_t / alpha_prod_t_prev
    
    # Make sure alpha_prod_t_prev is not zero if used in denominator
    # alpha_t = alpha_prod_t / alpha_prod_t_prev if t > 1 else alpha_prod_t (assuming alpha_prod for t=0 is 1.0)
    # Let's use the notebook notation:
    # a = alpha_prod[t-1] / alpha_prod[t-2] (current alpha_prod / prev alpha_prod)
    # ap = alpha_prod[t-1] (current alpha_prod)
    # ap_prev = alpha_prod[t-2] (prev alpha_prod)

    # current_alpha_t corresponds to alpha_prod_t / alpha_prod_t_prev
    # Need to handle t=1 where alpha_prod_t_prev (for t=0) would be 1.
    if t_tensor[0].item() > 1:
        # Ensure alpha_prod_t_prev is not zero, though for valid schedules it shouldn't be.
        # Add a small epsilon if there's a risk, but usually not needed for alpha_bar.
        safe_alpha_prod_t_prev = torch.max(alpha_prod_t_prev, torch.tensor(1e-8, device=device)) # prevent division by zero
        alpha_t = alpha_prod_t / safe_alpha_prod_t_prev
    else: # t=1
        alpha_t = alpha_prod_t # alpha_prod_t_prev for t=0 is 1.0

    # Ensure alpha_t is not exactly 0 to prevent division by zero in coeff_x_t
    # And alpha_prod_t is not exactly 1 to prevent division by zero in coeff_eps
    safe_alpha_t = torch.max(alpha_t, torch.tensor(1e-8, device=device))
    safe_one_minus_alpha_prod_t = torch.max(1.0 - alpha_prod_t, torch.tensor(1e-8, device=device))


    coeff_x_t = 1.0 / torch.sqrt(safe_alpha_t)
    coeff_eps = (1.0 - alpha_t) / torch.sqrt(safe_one_minus_alpha_prod_t)
    
    x_t_minus_1_mean = coeff_x_t * (x_t - coeff_eps * eps_theta)

    # Add noise (variance term) if t > 1
    if t_tensor[0].item() > 1:
        # sigma_t^2 = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_t)
        # This is beta_tilde_t from DDPM paper.
        # Or simply (1 - alpha_t) if using fixed variance to beta_t.
        # Notebook uses: sigma = torch.sqrt(((1.0 - ap_prev) / (1.0 - ap)) * (1.0 - a))
        
        # variance = ((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)) * (1.0 - alpha_t)
        # The above formula for variance can lead to issues if (1.0 - alpha_prod_t) is zero or very small.
        # Let's use the beta_tilde_t logic, but ensure no negative sqrt.
        # beta_t = 1 - alpha_t
        # beta_tilde_t = ( (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) ) * beta_t
        # We need to ensure beta_tilde_t is non-negative. Given schedules, it should be.

        beta_t = 1.0 - alpha_t # This alpha_t is (alpha_prod_t / alpha_prod_t_prev)
        variance_term_numerator = 1.0 - alpha_prod_t_prev
        variance_term_denominator = 1.0 - alpha_prod_t
        
        # Ensure denominator is not zero
        safe_variance_term_denominator = torch.max(variance_term_denominator, torch.tensor(1e-8, device=device))
        
        variance = (variance_term_numerator / safe_variance_term_denominator) * beta_t
        
        # Clip variance to be non-negative just in case of numerical precision issues
        sigma_t = torch.sqrt(torch.clamp(variance, min=1e-8)) # Ensure variance is positive before sqrt
        
        noise = torch.randn_like(x_t, device=device)
        x_t_minus_1 = x_t_minus_1_mean + sigma_t * noise
    else: # t=1, no noise added in DDPM sampling
        x_t_minus_1 = x_t_minus_1_mean
        
    return x_t_minus_1


def generate_samples_from_diffusion(num_samples, latent_shape):
    print(f"Using device: {cd.DEVICE}")

    # --- Load Diffusion Model ---
    print("Loading trained diffusion model...")
    diffusion_model = TimeEmbeddingModel(
        unet_features=cd.UNET_FEATURES,
        num_input_channels=cd.NUM_CHANNELS_LATENT,
        time_embedding_dim=cd.TIME_EMBEDDING_DIM,
        device=cd.DEVICE
    ).to(cd.DEVICE)
    try:
        diffusion_model.load_state_dict(torch.load(cd.DIFFUSION_MODEL_SAVE_PATH, map_location=cd.DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Diffusion model not found at {cd.DIFFUSION_MODEL_SAVE_PATH}. Train it first.")
        return
    except Exception as e:
        print(f"ERROR: Could not load diffusion model: {e}")
        return
    diffusion_model.eval()
    print("Diffusion model loaded.")

    # --- Load VAE Decoder ---
    print("Loading VAE model for its decoder...")
    # Load the full VAE model to use its decoder part
    vae_model = VAE(
        encoder_base_features=config_vae.FEATURES_ENC, # From original VAE config
        latent_dim_channels=config_vae.LATENT_DIM_CHANNELS
    ).to(cd.DEVICE)
    try:
        vae_model.load_state_dict(torch.load(cd.VAE_MODEL_PATH, map_location=cd.DEVICE))
    except FileNotFoundError:
        print(f"ERROR: VAE model not found at {cd.VAE_MODEL_PATH}. Train/place it first.")
        return
    except Exception as e:
        print(f"ERROR: Could not load VAE model: {e}")
        return
    vae_decoder = vae_model.decoder_model # Get the decoder part
    vae_decoder.eval()
    print("VAE decoder loaded.")

    # --- Get Alpha Product Schedule ---
    alpha_prod_schedule = get_alpha_prod(
        time_steps=cd.TIME_STEPS,
        schedule_type=cd.SCHEDULE_TYPE,
        linear_start=cd.LINEAR_SCHEDULE_START,
        linear_end=cd.LINEAR_SCHEDULE_END,
        cosine_s=cd.COSINE_SCHEDULE_S,
        device=cd.DEVICE
    )

    # --- Sampling Process (Reverse Diffusion) ---
    print(f"Starting sampling for {num_samples} images...")
    # Start with random noise (x_T)
    # Latent shape: (C, H, W) -> (num_samples, C, H, W)
    current_latents = torch.randn((num_samples,) + latent_shape, device=cd.DEVICE)

    for t_int_val in tqdm(range(cd.TIME_STEPS, 0, -1), desc="Reverse diffusion steps"): # Iterate with integer t
        # Prepare t as a tensor for the model
        t_tensor = torch.full((num_samples,), t_int_val, device=cd.DEVICE, dtype=torch.long)
        
        # Get alpha_bar_t and alpha_bar_{t-1}
        # alpha_prod_schedule is 0-indexed for t=1...T
        alpha_prod_t_val = alpha_prod_schedule[t_int_val-1] # For current time t
        alpha_prod_t_prev_val = alpha_prod_schedule[t_int_val-2] if t_int_val > 1 else torch.tensor(1.0, device=cd.DEVICE) # For t-1 (alpha_prod_0 = 1.0)
                                                                                             # alpha_prod_schedule[0] is alpha_bar_1

        # Reshape for broadcasting if needed by denoise_at_t
        # These are scalar values from the schedule, they need to be broadcastable to [B,1,1,1] if ops need it
        # Inside denoise_at_t, they are used with .view(-1,1,1,1) if not already shaped.
        # Let's pass them as scalars for now and let denoise_at_t handle shaping if needed by specific ops.
        # However, the way they are used directly in arithmetic with tensors x_t, eps_theta, etc. means they
        # likely need to be shaped correctly. Let's reshape here.
        
        current_alpha_prod = alpha_prod_t_val.view(-1,1,1,1) # Make it [1,1,1,1] then it broadcasts
        prev_alpha_prod = alpha_prod_t_prev_val.view(-1,1,1,1)


        with torch.no_grad():
            current_latents = denoise_at_t(
                diffusion_model, current_latents, t_tensor,
                current_alpha_prod, prev_alpha_prod, cd.DEVICE # Pass shaped tensors
            )
    
    print("Reverse diffusion complete. Generated latents.")
    generated_latents = current_latents # These are x_0_hat

    # --- Decode Latents to Images using VAE Decoder ---
    print("Decoding latents to images...")
    with torch.no_grad():
        generated_images_decoded = vae_decoder(generated_latents)
    
    # Clip images to [0, 1] as VAE was trained on this range
    generated_images_clipped = torch.clip(generated_images_decoded, 0.0, 1.0)
    print("Images decoded and clipped.")

    # --- Plot and Save Generated Images ---
    save_figure_path = os.path.join(cd.FIGURES_SAVE_DIR, "diffusion_generated_samples.png")
    plot_images(
        generated_images_clipped.cpu(),
        title=f"Generated Samples ({num_samples}, {cd.VAE_LATENT_H*8}x{cd.VAE_LATENT_W*8}) via Latent Diffusion",
        save_path=save_figure_path,
        n_row=cd.N_ROW_PLOT
    )
    print(f"Generated images plot saved to {save_figure_path}")

if __name__ == '__main__':
    # Define the shape of the latent variables your VAE produces
    # (Channels, Height, Width) - from your config_vae.py
    latent_c = cd.VAE_LATENT_DIM_CHANNELS
    latent_h = cd.VAE_LATENT_H
    latent_w = cd.VAE_LATENT_W
    shape_of_latents = (latent_c, latent_h, latent_w)

    generate_samples_from_diffusion(
        num_samples=cd.NUM_SAMPLES_TO_GENERATE,
        latent_shape=shape_of_latents
    )