import torch
import torch.optim as optim
import os
import math
from itertools import chain # For optimizer if using separate encoder/decoder

from config_vae import (DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, I_LOG,
                        FEATURES_ENC, LATENT_DIM_CHANNELS, MODEL_SAVE_PATH, RESULTS_DIR)
from model_vae import VAE, vae_loss_function # Encoder, Decoder classes removed if VAE encapsulates them
from data_utils import get_mnist_dataloaders
from plot_utils import plot_losses, plot_images
import torchvision.transforms.functional as TF # For potential cropping if needed for viz

def train_vae():
    print(f"Using device: {DEVICE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(batch_size=BATCH_SIZE)

    model = VAE(
        encoder_base_features=FEATURES_ENC,
        latent_dim_channels=LATENT_DIM_CHANNELS
    ).to(DEVICE)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}") # Should be ~50k with new config

    # Optimizer for the combined VAE model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'train_kl': [] # train_kl is the -D_KL term from notebook
    }

    fixed_test_images, _ = next(iter(test_loader))
    # Input images are 32x32. Visualize them as is.
    fixed_test_images_vis = fixed_test_images[:16].to(DEVICE)
    plot_images(fixed_test_images_vis, title="Original Test Images (32x32 Fixed Batch)",
                save_path=os.path.join(RESULTS_DIR, "original_test_images.png"), n_row=4)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        total_train_recon_loss = 0
        total_train_kl_div = 0 # This will store the negative KL div sum

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE) # Shape [B, 1, 32, 32]
            optimizer.zero_grad()
            
            recon_batch, mu, sigma_kl = model(data) # recon_batch is [B, 1, 32, 32]
            # No clipping before loss as per notebook's training logic
            
            loss, recon_loss_mean, kl_div_mean = vae_loss_function(recon_batch, data, mu, sigma_kl)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss_mean.item()
            total_train_kl_div += kl_div_mean.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_kl_div = total_train_kl_div / len(train_loader) # This is avg(-D_KL)
        
        if (epoch + 1) % I_LOG == 0 or epoch == EPOCHS -1 :
            model.eval()
            total_val_loss = 0
            # total_val_recon_loss = 0
            # total_val_kl_div = 0 # Not averaged for val_loss printout in notebook
            with torch.no_grad():
                # For consistent validation loss, use a loop like training
                val_loss_accum = 0.0
                val_recon_accum = 0.0
                val_kl_accum = 0.0
                for data_val, _ in test_loader:
                    data_val = data_val.to(DEVICE)
                    recon_val, mu_val, sigma_val_kl = model(data_val)
                    # No clipping before loss
                    v_loss, v_recon, v_kl = vae_loss_function(recon_val, data_val, mu_val, sigma_val_kl)
                    val_loss_accum += v_loss.item()
                    val_recon_accum += v_recon.item() # for detailed printing if desired
                    val_kl_accum += v_kl.item() # for detailed printing if desired

                avg_val_loss = val_loss_accum / len(test_loader)
                avg_val_recon = val_recon_accum / len(test_loader) # Optional
                avg_val_kl = val_kl_accum / len(test_loader) # Optional

                # Plot reconstructions of fixed test images
                recon_fixed, _, _ = model(fixed_test_images_vis)
                recon_fixed_clipped = torch.clip(recon_fixed, 0.0, 1.0) # Clip for visualization

                # Create a comparison image (original vs reconstructed)
                # Ensure both are on CPU for plotting if fixed_test_images_vis was on device
                comparison_imgs = []
                for i in range(fixed_test_images_vis.size(0)):
                    comparison_imgs.append(fixed_test_images_vis[i].cpu())
                    comparison_imgs.append(recon_fixed_clipped[i].cpu())
                comparison_grid = torch.stack(comparison_imgs)

                plot_images(comparison_grid, title=f"Reconstructions Epoch {epoch+1} (Orig | Recon)",
                            save_path=os.path.join(RESULTS_DIR, f"reconstruction_epoch_{epoch+1}.png"), n_row=8) # 8 pairs = 16 images


            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_recon'].append(avg_train_recon_loss)
            history['train_kl'].append(avg_train_kl_div) # avg_train_kl_div is E[-D_KL]

            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, KL_term: {avg_train_kl_div:.4f}) "
                  f"Val Loss: {avg_val_loss:.4f} (Val Recon: {avg_val_recon:.4f}, Val KL_term: {avg_val_kl:.4f})")
            
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    plot_losses(history['train_loss'], history['val_loss'],
                history['train_recon'], history['train_kl'], # train_kl is E[-D_KL]
                save_path=os.path.join(RESULTS_DIR, "loss_curves.png"))

if __name__ == '__main__':
    train_vae()