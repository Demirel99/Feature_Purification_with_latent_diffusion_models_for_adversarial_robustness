import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model and util imports
from model_vae import VAE
from diffusion_model import TimeEmbeddingModel
from diffusion_utils import get_alpha_prod
from data_utils import get_mnist_dataloaders
# plot_utils.plot_images is not used here, custom plotting is implemented.
import config_vae as cv
import config_diffusion as cd

# --- Helper function: denoise_at_t ---
# This function is copied from generate_diffusion_samples.py
# It performs one step of the reverse diffusion process.
def denoise_at_t(model, x_t, t_tensor, alpha_prod_t, alpha_prod_t_prev, device):
    """
    Performs one step of the reverse diffusion process.
    model: The trained U-Net diffusion model.
    x_t: Current noised sample at step t. [B, C, H, W]
    t_tensor: Current time step t, as a tensor for the model. [B]
    alpha_prod_t: $\bar{\alpha}_t$ (shaped for broadcasting, e.g., [B,1,1,1] or [1,1,1,1])
    alpha_prod_t_prev: $\bar{\alpha}_{t-1}$ (shaped for broadcasting)
    """
    if t_tensor[0].item() == 1:
        # For t=1, noise is not added in DDPM sampling.
        pass

    # Predict noise using U-Net
    eps_theta = model(x_t, t_tensor) # t_tensor should be (B,)

    # Calculate x_{t-1} mean component
    # alpha_t = alpha_prod_t / alpha_prod_t_prev
    if t_tensor[0].item() > 1:
        safe_alpha_prod_t_prev = torch.max(alpha_prod_t_prev, torch.tensor(1e-8, device=device))
        alpha_t = alpha_prod_t / safe_alpha_prod_t_prev
    else: # t=1
        alpha_t = alpha_prod_t # (since alpha_prod_t_prev for t=0 is 1.0)

    safe_alpha_t = torch.max(alpha_t, torch.tensor(1e-8, device=device))
    safe_one_minus_alpha_prod_t = torch.max(1.0 - alpha_prod_t, torch.tensor(1e-8, device=device))

    coeff_x_t = 1.0 / torch.sqrt(safe_alpha_t)
    coeff_eps = (1.0 - alpha_t) / torch.sqrt(safe_one_minus_alpha_prod_t)
    
    x_t_minus_1_mean = coeff_x_t * (x_t - coeff_eps * eps_theta)

    # Add noise (variance term) if t > 1
    if t_tensor[0].item() > 1:
        beta_t = 1.0 - alpha_t # This alpha_t is (alpha_prod_t / alpha_prod_t_prev)
        variance_term_numerator = 1.0 - alpha_prod_t_prev
        variance_term_denominator = 1.0 - alpha_prod_t
        
        safe_variance_term_denominator = torch.max(variance_term_denominator, torch.tensor(1e-8, device=device))
        variance = (variance_term_numerator / safe_variance_term_denominator) * beta_t
        
        # Clip variance to be non-negative
        sigma_t = torch.sqrt(torch.clamp(variance, min=1e-8))
        
        noise = torch.randn_like(x_t, device=device)
        x_t_minus_1 = x_t_minus_1_mean + sigma_t * noise
    else: # t=1, no noise added in DDPM sampling
        x_t_minus_1 = x_t_minus_1_mean
        
    return x_t_minus_1


def run_partial_diffusion_test(num_images=4, n_partial_steps=10):
    print(f"--- Partial Diffusion Test ---")
    print(f"Using device: {cd.DEVICE}")
    print(f"Number of images to test: {num_images}")
    print(f"Partial diffusion steps (forward & reverse): {n_partial_steps}")

    # --- Load VAE Model ---
    print("\nLoading VAE model...")
    vae_model = VAE(
        encoder_base_features=cv.FEATURES_ENC,
        latent_dim_channels=cv.LATENT_DIM_CHANNELS
    ).to(cd.DEVICE)
    try:
        vae_model.load_state_dict(torch.load(cd.VAE_MODEL_PATH, map_location=cd.DEVICE))
    except FileNotFoundError:
        print(f"ERROR: VAE model not found at {cd.VAE_MODEL_PATH}. Train/place it first.")
        return
    except Exception as e:
        print(f"ERROR: Could not load VAE model: {e}")
        return
    vae_model.eval()
    vae_encoder = vae_model.encoder_model
    vae_decoder = vae_model.decoder_model
    print("VAE model loaded successfully.")

    # --- Load Diffusion Model ---
    print("\nLoading trained diffusion model...")
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
    print("Diffusion model loaded successfully.")

    # --- Get Alpha Product Schedule ---
    alpha_prod_schedule = get_alpha_prod(
        time_steps=cd.TIME_STEPS,
        schedule_type=cd.SCHEDULE_TYPE,
        linear_start=cd.LINEAR_SCHEDULE_START,
        linear_end=cd.LINEAR_SCHEDULE_END,
        cosine_s=cd.COSINE_SCHEDULE_S,
        device=cd.DEVICE
    )

    # --- Load Data ---
    print("\nLoading MNIST test data...")
    # We need a small batch_size for num_images
    _, test_loader = get_mnist_dataloaders(batch_size=num_images, root='./data')
    input_images_batch, _ = next(iter(test_loader)) # Get one batch
    input_images = input_images_batch.to(cd.DEVICE) # Ensure it's on the correct device
    print(f"Loaded {input_images.shape[0]} test images.")

    # --- 1. VAE Encode input images to get z_0 ---
    print("\nStep 1: Encoding input images with VAE encoder...")
    with torch.no_grad():
        mu_sigma_map = vae_encoder(input_images)
        mu = mu_sigma_map[:, :cv.LATENT_DIM_CHANNELS, :, :]
        sigma_params = mu_sigma_map[:, cv.LATENT_DIM_CHANNELS:, :, :]
        z_0 = vae_model.reparameterize(mu, sigma_params) # Initial latents
    print(f"  Encoded latents (z_0) shape: {z_0.shape}")

    # --- Get VAE direct reconstruction for comparison ---
    with torch.no_grad():
        reconstructed_images_vae_direct = torch.clip(vae_decoder(z_0), 0.0, 1.0)

    # --- 2. Partial Forward Diffusion: Noise z_0 to z_N_PARTIAL ---
    print(f"\nStep 2: Applying {n_partial_steps} analytical forward diffusion steps to z_0...")
    t_target = n_partial_steps
    if not (1 <= t_target <= cd.TIME_STEPS):
        print(f"ERROR: n_partial_steps ({t_target}) must be between 1 and {cd.TIME_STEPS}. Exiting.")
        return
    
    # alpha_prod_schedule is 0-indexed (for t=1 to T, indices are 0 to T-1)
    alpha_prod_t_target = alpha_prod_schedule[t_target - 1] 
    
    sqrt_alpha_prod_t_target = torch.sqrt(alpha_prod_t_target)
    sqrt_one_minus_alpha_prod_t_target = torch.sqrt(1.0 - alpha_prod_t_target)
    
    epsilon_noise = torch.randn_like(z_0, device=cd.DEVICE)
    with torch.no_grad():
        z_n_partial = sqrt_alpha_prod_t_target * z_0 + sqrt_one_minus_alpha_prod_t_target * epsilon_noise
    print(f"  Noised latents (z_{n_partial_steps}) shape: {z_n_partial.shape}")

    # --- 3. Partial Reverse Diffusion: Denoise z_N_PARTIAL back to an estimate of z_0 ---
    print(f"\nStep 3: Applying {n_partial_steps} reverse diffusion steps from z_{n_partial_steps} using U-Net...")
    current_latents_for_reverse = z_n_partial.clone() 

    for t_int_val in tqdm(range(n_partial_steps, 0, -1), desc="Partial reverse diffusion"):
        # Prepare t_tensor for the diffusion model
        t_tensor = torch.full((current_latents_for_reverse.shape[0],), t_int_val, device=cd.DEVICE, dtype=torch.long)
        
        # Get alpha_bar_t and alpha_bar_{t-1}
        alpha_prod_t_val = alpha_prod_schedule[t_int_val - 1]
        alpha_prod_t_prev_val = alpha_prod_schedule[t_int_val - 2] if t_int_val > 1 else torch.tensor(1.0, device=cd.DEVICE)
        
        # Reshape alpha products for broadcasting within denoise_at_t
        current_alpha_prod_reshaped = alpha_prod_t_val.view(-1, 1, 1, 1) 
        prev_alpha_prod_reshaped = alpha_prod_t_prev_val.view(-1, 1, 1, 1)

        with torch.no_grad():
            current_latents_for_reverse = denoise_at_t(
                diffusion_model, 
                current_latents_for_reverse, 
                t_tensor,
                current_alpha_prod_reshaped, 
                prev_alpha_prod_reshaped, 
                cd.DEVICE
            )
    z_0_reconstructed_via_diffusion = current_latents_for_reverse
    print(f"  Denoised latents (estimate of z_0) shape: {z_0_reconstructed_via_diffusion.shape}")

    # --- 4. VAE Decode the reconstructed latents ---
    print("\nStep 4: Decoding the reconstructed latents with VAE decoder...")
    with torch.no_grad():
        reconstructed_images_diffusion = torch.clip(vae_decoder(z_0_reconstructed_via_diffusion), 0.0, 1.0)
    print(f"  Final reconstructed images shape via partial diffusion: {reconstructed_images_diffusion.shape}")

    # --- 5. Visualization ---
    print("\nStep 5: Plotting results...")
    
    input_images_cpu = input_images.cpu()
    reconstructed_images_vae_direct_cpu = reconstructed_images_vae_direct.cpu()
    reconstructed_images_diffusion_cpu = reconstructed_images_diffusion.cpu()

    fig, axes = plt.subplots(num_images, 3, figsize=(9, num_images * 3)) # width, height
    if num_images == 1: # Ensure axes is always 2D for consistent indexing
        axes = np.array([axes])

    fig.suptitle(f"Partial Diffusion Test ({n_partial_steps} steps)", fontsize=16)
    
    for i in range(num_images):
        axes[i, 0].imshow(input_images_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructed_images_vae_direct_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title("VAE Recon (Enc->Dec)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(reconstructed_images_diffusion_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title(f"Diff Recon (Enc->z{n_partial_steps}->Dec)")
        axes[i, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    
    # Ensure figures directory exists
    os.makedirs(cd.FIGURES_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(cd.FIGURES_SAVE_DIR, f"partial_diffusion_test_{n_partial_steps}_steps_imgs_{num_images}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()
    print("\n--- Test Complete ---")

if __name__ == '__main__':
    # You can change these parameters for testing:
    NUM_IMAGES_TO_TEST = 4  # How many images to process and display
    N_PARTIAL_DIFF_STEPS = 100 # Number of noising steps, and then number of denoising steps

    run_partial_diffusion_test(num_images=NUM_IMAGES_TO_TEST, n_partial_steps=N_PARTIAL_DIFF_STEPS)
    
    # Example: Test with only 1 step (maximum information from z_0, minimal diffusion effect)
    # run_partial_diffusion_test(num_images=2, n_partial_steps=1)

    # Example: Test with more steps (more noising, relying more on diffusion model)
    # run_partial_diffusion_test(num_images=2, n_partial_steps=50)