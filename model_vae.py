import torch
import torch.nn as nn
import torch.nn.functional as F
# config_vae imports are used in train/generate scripts directly

class ConvBlock(torch.nn.Module):
    def __init__(self, fin, fout, kernel_size=3, padding='same', *args, **kwargs):
        super(ConvBlock, self).__init__()
        # The notebook uses default arguments for Conv2d, which include bias=True
        self._conv = torch.nn.Conv2d(fin, fout, kernel_size=kernel_size, padding=padding, *args, **kwargs)
        self._norm = torch.nn.BatchNorm2d(fout)
        self._relu = torch.nn.LeakyReLU() # Defaults to 0.01 negative slope

    def forward(self, x):
        return self._relu(self._norm(self._conv(x)))

class ConvEncoder(torch.nn.Module):
    # features_list example for VAE: [1, 8, 16, 32, 2*LATENT_DIM_CHANNELS]
    def __init__(self, features_list):
        super(ConvEncoder, self).__init__()
        layers = []
        for i in range(len(features_list) - 1):
            fi = features_list[i]
            fo = features_list[i+1]
            if i > 0: # MaxPool for all but the first layer transition
                layers.append(torch.nn.Sequential(
                    torch.nn.MaxPool2d(2),
                    ConvBlock(fi, fo, 3, padding='same'),
                    ConvBlock(fo, fo, 3, padding='same'),
                ))
            else: # First layer transition, no MaxPool
                layers.append(torch.nn.Sequential(
                    ConvBlock(fi, fo, 3, padding='same'),
                    ConvBlock(fo, fo, 3, padding='same'),
                ))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class ConvDecoder(torch.nn.Module):
    # features_list example for VAE: [LATENT_DIM_CHANNELS, 32, 16, 8, 1]
    def __init__(self, features_list):
        super(ConvDecoder, self).__init__()
        layers = []
        num_transitions = len(features_list) - 1

        for i in range(num_transitions):
            layer_ops = []
            fi = features_list[i]
            fo = features_list[i+1]

            if i > 0:
                layer_ops += [
                    # Notebook uses default mode='nearest' for Upsample
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    ConvBlock(fi, fi, 3, padding='same'),
                ]

            if i < num_transitions - 1: # Not the last transition
                layer_ops += [
                    ConvBlock(fi, fi, 3, padding='same'),
                    ConvBlock(fi, fo, 3, padding='same'),
                ]
            else: # Last transition to output image
                layer_ops += [
                    ConvBlock(fi, fi, 3, padding='same'),
                    # Notebook uses kernel_size=3 for the final Conv2d
                    torch.nn.Conv2d(fi, fo, 3, padding='same'),
                    # No Sigmoid, clipping is done outside
                ]
            layers.append(torch.nn.Sequential(*layer_ops))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class VAE(nn.Module):
    def __init__(self, encoder_base_features, latent_dim_channels):
        super(VAE, self).__init__()
        self.latent_dim_channels = latent_dim_channels

        # Encoder: encoder_base_features + [2 * latent_dim_channels]
        # Example: [1, 8, 16, 32] + [2*2] -> [1, 8, 16, 32, 4]
        self.encoder_model = ConvEncoder(encoder_base_features + [2 * self.latent_dim_channels])

        # Decoder: [latent_dim_channels] + reversed_base_features_and_output_channel
        # Example: [2] + [32, 16, 8, 1] (original input channel 1 becomes output channel 1)
        decoder_feature_list = [self.latent_dim_channels] + encoder_base_features[:0:-1] + [encoder_base_features[0]]
        self.decoder_model = ConvDecoder(decoder_feature_list)

    def reparameterize(self, mu, sigma_maybe_logvar):
        # Notebook directly uses sigma output. For stability:
        # Assume sigma_maybe_logvar is params for std_dev, apply softplus.
        std = F.softplus(sigma_maybe_logvar) + 1e-6 # Ensure positivity and avoid collapse
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_sigma_map = self.encoder_model(x)
        
        mu = mu_sigma_map[:, :self.latent_dim_channels, :, :]
        sigma_params = mu_sigma_map[:, self.latent_dim_channels:, :, :]
        
        # sigma for reparameterize should be std_dev and positive.
        # LeakyReLU output (sigma_params) can be negative.
        # The notebook uses sigma_params directly as sigma (std_dev).
        # This is risky for log(sigma^2) if sigma can be 0.
        # I'll use reparameterize with softplus for stability.
        z = self.reparameterize(mu, sigma_params) # sigma_params are used to derive std
        
        reconstructed_x = self.decoder_model(z)
        
        # The 'sigma' returned for KL divergence should match what the KL formula expects.
        # The KL formula L = sigma^2 + mu^2 - log(sigma^2) - 1 implies sigma is std_dev.
        # We use the stable std from reparameterize for KL calculation.
        sigma_for_kl = F.softplus(sigma_params) + 1e-6
        return reconstructed_x, mu, sigma_for_kl


def vae_loss_function(recon_x, x, mu, sigma_kl):
    # Reconstruction loss (MSE as per article/notebook)
    # Sum over spatial and channel dimensions, then mean over batch
    recon_loss_elementwise = F.mse_loss(recon_x, x, reduction='none')
    recon_loss = torch.sum(recon_loss_elementwise, dim=[1, 2, 3]) # Sum over C, H, W
    
    # KL divergence: sigma_kl is the standard deviation (must be > 0)
    sigma2 = sigma_kl.pow(2)
    # Add a small epsilon to log argument to prevent log(0)
    kl_term_elementwise = sigma2 + mu.pow(2) - torch.log(sigma2 + 1e-8) - 1.0
    # Notebook: kl_loss = -0.5 * torch.sum(...)
    kl_div_summed = -0.5 * torch.sum(kl_term_elementwise, dim=[1, 2, 3]) # Sum over C, H, W

    # Notebook: loss = reconstruction_loss - kl_loss (where kl_loss is already -D_KL)
    # So, total_loss = torch.mean(recon_loss - kl_div_summed)
    # This is equivalent to: torch.mean(recon_loss + D_KL_positive)
    total_loss = torch.mean(recon_loss - kl_div_summed)
    
    return total_loss, torch.mean(recon_loss), torch.mean(kl_div_summed)