import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Diffusion model components
from diffusion_model import TimeEmbeddingModel
from diffusion_utils import get_alpha_prod
import config_diffusion as cd

# User's plot_utils
from plot_utils import plot_images # If needed for visualizing latents/etc.
                                   # A new plot_diffusion_losses might be good.

def plot_diffusion_losses(train_losses, val_losses, save_path=None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Avg Train Diffusion Loss')
    plt.plot(epochs_range, val_losses, label='Avg Val Diffusion Loss')
    plt.xlabel(f'Epochs (x{cd.I_LOG_DIFFUSION})')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Diffusion Model Training Losses')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved diffusion loss plot to {save_path}")
    plt.show()


def do_diffusion_step(model, x_0, t, alpha_prod_tensor, device):
    # x_0: batch of original latents [B, C, H, W]
    # t: batch of time steps [B]
    # alpha_prod_tensor: precomputed alpha_bar_t values [TIME_STEPS]
    
    # Sample noise
    eps_noise = torch.randn_like(x_0, device=device)

    # Get alpha_bar_t for the sampled time steps t
    # alpha_prod_tensor is 0-indexed (for t=1 to T, indices are 0 to T-1)
    # So for time t, use index t-1
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_tensor[t-1]).view(-1, 1, 1, 1) # Reshape for broadcasting
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_tensor[t-1]).view(-1, 1, 1, 1)

    # Calculate x_t (noised latents)
    x_t = sqrt_alpha_prod_t * x_0 + sqrt_one_minus_alpha_prod_t * eps_noise
    
    # Predict noise using the U-Net model
    eps_theta_predicted = model(x_t, t) # model expects t as (B,)
    
    # Calculate MSE loss between actual noise and predicted noise
    loss = F.mse_loss(eps_noise, eps_theta_predicted)
    return loss

def train_diffusion_model():
    print(f"Using device: {cd.DEVICE}")

    # --- Load Pre-generated Latents ---
    try:
        latents_train = torch.load(cd.LATENT_TRAIN_PATH)
        latents_val = torch.load(cd.LATENT_VAL_PATH)
        print(f"Loaded training latents: {latents_train.shape}")
        print(f"Loaded validation latents: {latents_val.shape}")
    except FileNotFoundError:
        print("ERROR: Latent files not found. Please run generate_latents.py first.")
        return

    train_dataset = TensorDataset(latents_train)
    val_dataset = TensorDataset(latents_val)
    train_loader = DataLoader(train_dataset, batch_size=cd.BATCH_SIZE_DIFFUSION, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cd.BATCH_SIZE_DIFFUSION, shuffle=False, num_workers=2, pin_memory=True)

    # --- Initialize Model, Optimizer, Schedule ---
    diffusion_model = TimeEmbeddingModel(
        unet_features=cd.UNET_FEATURES,
        num_input_channels=cd.NUM_CHANNELS_LATENT, # Should match VAE's latent_dim_channels
        time_embedding_dim=cd.TIME_EMBEDDING_DIM,
        device=cd.DEVICE # Pass device to SinusoidalPositionEmbedding
    ).to(cd.DEVICE)
    
    num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Diffusion U-Net initialized with {num_params} parameters.")

    optimizer = optim.Adam(diffusion_model.parameters(), lr=cd.LEARNING_RATE_DIFFUSION)
    
    alpha_prod_schedule = get_alpha_prod(
        time_steps=cd.TIME_STEPS,
        schedule_type=cd.SCHEDULE_TYPE,
        linear_start=cd.LINEAR_SCHEDULE_START,
        linear_end=cd.LINEAR_SCHEDULE_END,
        cosine_s=cd.COSINE_SCHEDULE_S,
        device=cd.DEVICE
    )

    history = {'train_loss': [], 'val_loss': []}

    # --- Training Loop ---
    print(f"Starting diffusion model training for {cd.EPOCHS_DIFFUSION} epochs...")
    for epoch in range(1, cd.EPOCHS_DIFFUSION + 1):
        diffusion_model.train()
        total_epoch_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cd.EPOCHS_DIFFUSION} [Training]")
        for (batch_latents,) in progress_bar: # batch_latents is x_0
            x_0 = batch_latents.to(cd.DEVICE)
            
            # Sample time steps t uniformly for each latent in the batch
            # t values should be from 1 to TIME_STEPS inclusive
            t = torch.randint(1, cd.TIME_STEPS + 1, (x_0.shape[0],), device=cd.DEVICE)
            
            optimizer.zero_grad()
            loss = do_diffusion_step(diffusion_model, x_0, t, alpha_prod_schedule, cd.DEVICE)
            loss.backward()
            optimizer.step()
            
            total_epoch_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_train_loss = total_epoch_train_loss / len(train_loader)

        # --- Validation ---
        if epoch % cd.I_LOG_DIFFUSION == 0 or epoch == cd.EPOCHS_DIFFUSION:
            diffusion_model.eval()
            total_epoch_val_loss = 0.0
            with torch.no_grad():
                val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{cd.EPOCHS_DIFFUSION} [Validation]")
                for (batch_latents_val,) in val_progress_bar:
                    x_0_val = batch_latents_val.to(cd.DEVICE)
                    t_val = torch.randint(1, cd.TIME_STEPS + 1, (x_0_val.shape[0],), device=cd.DEVICE)
                    
                    val_loss = do_diffusion_step(diffusion_model, x_0_val, t_val, alpha_prod_schedule, cd.DEVICE)
                    total_epoch_val_loss += val_loss.item()
                    val_progress_bar.set_postfix(val_loss=val_loss.item())
            
            avg_epoch_val_loss = total_epoch_val_loss / len(val_loader)
            history['train_loss'].append(avg_epoch_train_loss)
            history['val_loss'].append(avg_epoch_val_loss)
            
            print(f"Epoch {epoch}/{cd.EPOCHS_DIFFUSION} | Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {avg_epoch_val_loss:.6f}")
            
            # Save model checkpoint
            torch.save(diffusion_model.state_dict(), cd.DIFFUSION_MODEL_SAVE_PATH)
            print(f"Diffusion model checkpoint saved to {cd.DIFFUSION_MODEL_SAVE_PATH}")

    print("Diffusion model training complete.")
    # Save final model
    torch.save(diffusion_model.state_dict(), cd.DIFFUSION_MODEL_SAVE_PATH)
    print(f"Final diffusion model saved to {cd.DIFFUSION_MODEL_SAVE_PATH}")

    # Plot losses
    loss_plot_path = os.path.join(cd.FIGURES_SAVE_DIR, "diffusion_loss_curves.png")
    plot_diffusion_losses(history['train_loss'], history['val_loss'], save_path=loss_plot_path)

if __name__ == '__main__':
    train_diffusion_model()