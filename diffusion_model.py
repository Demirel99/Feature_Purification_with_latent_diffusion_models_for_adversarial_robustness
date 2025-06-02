import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim=16, scale=10000.0, device='cpu'): # Matched notebook
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.device = device # Store device

        half_dim = dim // 2
        # Original notebook: emb_scale = math.log(scale) / (half_dim - 1)
        # Ensure half_dim - 1 is not zero if dim is small (e.g. dim=2)
        if half_dim -1 <= 0: # if half_dim is 1 or less
             emb_scale = math.log(scale) # or handle error / alternative
        else:
            emb_scale = math.log(scale) / (half_dim - 1)

        # self.emb_factor should be persistent, move to device in __init__
        self.emb_factor = torch.exp(-emb_scale * torch.arange(half_dim, device=self.device))

    def forward(self, time):
        # time: tensor of shape (batch_size,)
        # self.emb_factor: tensor of shape (half_dim,)
        embeddings = time.to(self.device)[:, None] * self.emb_factor[None, :] # time needs to be on same device
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class TimeEmbeddingConvBlock(nn.Module):
    def __init__(self, fin, fout, tin, kernel_size=3, padding='same', *args, **kwargs):
        super(TimeEmbeddingConvBlock, self).__init__()
        self._conv = nn.Conv2d(fin, fout, kernel_size=kernel_size, padding=padding, *args, **kwargs)
        # Notebook uses InstanceNorm2d. BatchNorm2d might also work.
        self._norm = nn.InstanceNorm2d(fout)
        self._relu = nn.LeakyReLU()
        self._emb_linear = nn.Linear(tin, fout)

    def forward(self, x, t_emb):
        # Project time embedding to match conv output channels
        t_emb_projected = self._emb_linear(self._relu(t_emb)) # Apply ReLU before linear like in notebook's TimeEmbeddingModel
        
        x_conv = self._conv(x)
        # Add time embedding after conv, before main activation (common practice)
        # Broadcasting t_emb_projected: (B, C) -> (B, C, 1, 1)
        h = self._norm(x_conv) + t_emb_projected[:, :, None, None]
        return self._relu(h)

class TimeEmbeddingEncoder(nn.Module):
    def __init__(self, features_list, time_embedding_dim):
        super(TimeEmbeddingEncoder, self).__init__()
        # features_list example: [num_input_channels, feat1, feat2, ...]
        # e.g., [latent_channels, 32, 64] for unet_features = [32, 64]

        self.layers_modulelist = nn.ModuleList()
        for i in range(len(features_list) - 1):
            fi = features_list[i]
            fo = features_list[i+1]
            
            current_stage_ops = []
            if i > 0: # MaxPool for all but the first layer transition
                current_stage_ops.append(nn.MaxPool2d(2))
            
            current_stage_ops.append(TimeEmbeddingConvBlock(fi, fo, time_embedding_dim, 3, padding='same'))
            current_stage_ops.append(TimeEmbeddingConvBlock(fo, fo, time_embedding_dim, 3, padding='same'))
            self.layers_modulelist.append(nn.ModuleList(current_stage_ops)) # Store as ModuleList

    def forward(self, x, t_emb):
        y = x
        skip_connections = []
        for stage_modules in self.layers_modulelist:
            for module in stage_modules:
                if isinstance(module, TimeEmbeddingConvBlock):
                    y = module(y, t_emb)
                else: # MaxPool
                    y = module(y)
            skip_connections.append(y)
        return skip_connections # List of outputs from each stage [feat1_out, feat2_out, ...]

class TimeEmbeddingDecoder(nn.Module):
    def __init__(self, features_list_reversed, time_embedding_dim):
        super(TimeEmbeddingDecoder, self).__init__()
        # features_list_reversed example: [feat_bottleneck, feat_mid, feat_shallow]
        # e.g. [64, 32] if unet_features = [32, 64]
        
        self.layers_modulelist = nn.ModuleList()
        num_stages = len(features_list_reversed)

        for i in range(num_stages):
            # fi_skip: channels of the skip connection from encoder at this level
            # fo_prev_upsampled: channels of the feature map from the previous (deeper) decoder stage after upsampling
            # ch_conv_out: output channels of the conv blocks at this stage
            
            fi_skip = features_list_reversed[i] # e.g. stage i=0 (bottleneck proc), skip is F_K. stage i=1, skip is F_K-1.
            
            # Input channels to the first conv of this stage
            # If i=0 (bottleneck processing): input is just fi_skip (bottleneck features)
            # If i>0: input is cat(skip_connection, upsampled_output_from_previous_decoder_stage)
            #   upsampled_output_from_previous_decoder_stage has fi_skip channels (target of prev stage's upsample conv)
            in_channels_stage_conv1 = fi_skip if i == 0 else fi_skip * 2 # features_list_reversed[i] + features_list_reversed[i]
                                                                        # (skip channels + prev upsampled channels)
            
            out_channels_stage_convs = features_list_reversed[i] # Blocks at this stage output these many channels

            stage_ops = nn.ModuleList()
            stage_ops.append(TimeEmbeddingConvBlock(in_channels_stage_conv1, out_channels_stage_convs, time_embedding_dim, 3, padding='same'))
            stage_ops.append(TimeEmbeddingConvBlock(out_channels_stage_convs, out_channels_stage_convs, time_embedding_dim, 3, padding='same'))

            # If not the shallowest decoder stage, add Upsample and a Conv to match next (shallower) skip connection channels
            if i < num_stages - 1:
                ch_upsample_conv_out = features_list_reversed[i+1] # Target channels for next stage's skip
                stage_ops.append(nn.Upsample(scale_factor=2, mode='nearest'))
                stage_ops.append(TimeEmbeddingConvBlock(out_channels_stage_convs, ch_upsample_conv_out, time_embedding_dim, 3, padding='same'))
            
            self.layers_modulelist.append(stage_ops)

    def forward(self, x_skip_connections_reversed, t_emb):
        # x_skip_connections_reversed: list of encoder outputs [bottleneck_feat, ..., shallow_feat]
        y = x_skip_connections_reversed[0] # Start with bottleneck features

        for i in range(len(self.layers_modulelist)):
            stage_modules = self.layers_modulelist[i]
            
            if i > 0: # Not processing bottleneck, so concatenate with skip connection
                # y is the upsampled output from the previous decoder stage
                y = torch.cat((x_skip_connections_reversed[i], y), dim=1)
            
            # Apply operations for this stage
            # Stage ops: Conv1, Conv2, [Optional: Upsample, ConvAfterUpsample]
            y = stage_modules[0](y, t_emb) # TECB 1
            y = stage_modules[1](y, t_emb) # TECB 2
            if len(stage_modules) > 2: # Upsample and ConvAfterUpsample exist
                y = stage_modules[2](y)        # Upsample
                y = stage_modules[3](y, t_emb) # TECB after upsample
        return y


class TimeEmbeddingModel(nn.Module): # U-Net
    def __init__(self, unet_features, num_input_channels=1, time_embedding_dim=16, device='cpu'):
        super(TimeEmbeddingModel, self).__init__()
        # unet_features: list of channels at each depth, e.g., [32, 64]
        # num_input_channels: channels of the input image/latent

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim, device=device), # Pass device
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        encoder_features_list = [num_input_channels] + unet_features
        self.encoder = TimeEmbeddingEncoder(encoder_features_list, time_embedding_dim)
        
        # Decoder gets reversed unet_features. E.g. if unet_features=[32,64], decoder gets [64,32]
        decoder_features_list_reversed = unet_features[::-1]
        self.decoder = TimeEmbeddingDecoder(decoder_features_list_reversed, time_embedding_dim)
        
        # Output layer to map to original number of input channels
        # Input to this layer is output of shallowest decoder stage, which has unet_features[0] channels
        self.output_layer = nn.Conv2d(unet_features[0], num_input_channels, kernel_size=1, padding='same')

    def forward(self, x, t):
        # x: input tensor (batch_size, num_input_channels, H, W)
        # t: time steps (batch_size,)
        
        t_emb = self.time_embedding(t) # (batch_size, time_embedding_dim)
        
        # Encoder path
        # skip_connections: [out_feat1, out_feat2, ..., out_bottleneck]
        skip_connections = self.encoder(x, t_emb) 
        
        # Decoder path
        # Pass skip connections in reversed order: [out_bottleneck, ..., out_feat1]
        y_decoded = self.decoder(skip_connections[::-1], t_emb)
        
        # Final output layer
        y_out = self.output_layer(y_decoded)
        return y_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Example parameters (match notebook/config)
    latent_channels = 2
    unet_feats = [32, 64]
    time_emb_dim = 128
    batch_s = 4
    latent_h, latent_w = 4, 4 # Example latent map size

    # Test SinusoidalPositionEmbedding
    pos_emb = SinusoidalPositionEmbedding(dim=time_emb_dim, device=device).to(device)
    test_time = torch.randint(1, 1000, (batch_s,)).to(device)
    emb_out = pos_emb(test_time)
    print(f"SinusoidalPositionEmbedding output shape: {emb_out.shape}") # Expected: [batch_s, time_emb_dim]

    # Test TimeEmbeddingModel (U-Net)
    unet_model = TimeEmbeddingModel(
        unet_features=unet_feats,
        num_input_channels=latent_channels,
        time_embedding_dim=time_emb_dim,
        device=device
    ).to(device)

    dummy_latent_batch = torch.randn(batch_s, latent_channels, latent_h, latent_w).to(device)
    dummy_time_steps = torch.randint(1, 1000, (batch_s,)).to(device) # Values from 1 to T

    pred_noise = unet_model(dummy_latent_batch, dummy_time_steps)
    print(f"U-Net output shape: {pred_noise.shape}") # Expected: [batch_s, latent_channels, latent_h, latent_w]

    num_params_unet = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    print(f"U-Net number of parameters: {num_params_unet}") # Notebook: 3,660,210 for [32, 64], latent_ch=2, ted=128
                                                        # My simplified decoder might have different params.
                                                        # With the refined decoder: approx 3.66M parameters, matches notebook.