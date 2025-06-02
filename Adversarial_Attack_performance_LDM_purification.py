# FILE: evaluate_purification_accuracy.py
import torch
import torch.nn.functional as F # For F.cross_entropy in FGSM
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model and util imports
from model_vae import VAE
from diffusion_model import TimeEmbeddingModel
from model import Classifier # For loading the pre-trained classifier
from diffusion_utils import get_alpha_prod
from data_utils import get_mnist_dataloaders
import config_vae as cv
import config_diffusion as cd

# Path for the pre-trained classifier model
CLASSIFIER_MODEL_PATH = 'mnist_classifier_weights.pth'

# --- FGSM Attack Function ---
def fgsm_attack(model, images, labels, epsilon, device):
    """
    Generates adversarial examples using the Fast Gradient Sign Method.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True

    outputs = model(images)
    model.zero_grad() # Important to zero gradients of the classifier
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    # Collect the gradients
    data_grad = images.grad.data
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * data_grad.sign()
    # Clip perturbed image to maintain the original data range (e.g., [0,1])
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.detach()


# --- Helper function: denoise_at_t (copied from previous version) ---
def denoise_at_t(model, x_t, t_tensor, alpha_prod_t, alpha_prod_t_prev, device):
    if t_tensor[0].item() == 1:
        pass
    eps_theta = model(x_t, t_tensor)
    if t_tensor[0].item() > 1:
        safe_alpha_prod_t_prev = torch.max(alpha_prod_t_prev, torch.tensor(1e-8, device=device))
        alpha_t = alpha_prod_t / safe_alpha_prod_t_prev
    else:
        alpha_t = alpha_prod_t
    safe_alpha_t = torch.max(alpha_t, torch.tensor(1e-8, device=device))
    safe_one_minus_alpha_prod_t = torch.max(1.0 - alpha_prod_t, torch.tensor(1e-8, device=device))
    coeff_x_t = 1.0 / torch.sqrt(safe_alpha_t)
    coeff_eps = (1.0 - alpha_t) / torch.sqrt(safe_one_minus_alpha_prod_t)
    x_t_minus_1_mean = coeff_x_t * (x_t - coeff_eps * eps_theta)
    if t_tensor[0].item() > 1:
        beta_t = 1.0 - alpha_t
        variance_term_numerator = 1.0 - alpha_prod_t_prev
        variance_term_denominator = 1.0 - alpha_prod_t
        safe_variance_term_denominator = torch.max(variance_term_denominator, torch.tensor(1e-8, device=device))
        variance = (variance_term_numerator / safe_variance_term_denominator) * beta_t
        sigma_t = torch.sqrt(torch.clamp(variance, min=1e-8))
        noise = torch.randn_like(x_t, device=device)
        x_t_minus_1 = x_t_minus_1_mean + sigma_t * noise
    else:
        x_t_minus_1 = x_t_minus_1_mean
    return x_t_minus_1

def get_classification_accuracy(classifier_model, images_batch, labels_batch, device):
    classifier_model.eval()
    images_batch = images_batch.to(device)
    labels_batch = labels_batch.to(device)
    with torch.no_grad():
        outputs = classifier_model(images_batch)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels_batch).sum().item()
        accuracy = correct / labels_batch.size(0)
    return accuracy


def run_purification_and_evaluate_accuracy(num_images=4, n_partial_steps=10, fgsm_epsilon=0.1):
    print(f"--- Latent Diffusion Purification and Accuracy Evaluation with FGSM Attack ---")
    print(f"Using device: {cd.DEVICE}")
    print(f"Number of images to test: {num_images}")
    print(f"Partial diffusion steps (forward & reverse): {n_partial_steps}")
    print(f"FGSM Epsilon: {fgsm_epsilon}")

    # --- Load Models ---
    print("\nLoading VAE model...")
    vae_model = VAE(encoder_base_features=cv.FEATURES_ENC, latent_dim_channels=cv.LATENT_DIM_CHANNELS).to(cd.DEVICE)
    try:
        vae_model.load_state_dict(torch.load(cd.VAE_MODEL_PATH, map_location=cd.DEVICE))
    except Exception as e: print(f"ERROR loading VAE: {e}"); return
    vae_model.eval()
    vae_encoder = vae_model.encoder_model
    vae_decoder = vae_model.decoder_model
    print("VAE model loaded.")

    print("\nLoading trained diffusion model...")
    diffusion_model = TimeEmbeddingModel(unet_features=cd.UNET_FEATURES, num_input_channels=cd.NUM_CHANNELS_LATENT, time_embedding_dim=cd.TIME_EMBEDDING_DIM, device=cd.DEVICE).to(cd.DEVICE)
    try:
        diffusion_model.load_state_dict(torch.load(cd.DIFFUSION_MODEL_SAVE_PATH, map_location=cd.DEVICE))
    except Exception as e: print(f"ERROR loading Diffusion Model: {e}"); return
    diffusion_model.eval()
    print("Diffusion model loaded.")

    print("\nLoading pre-trained classifier model...")
    classifier_model = Classifier().to(cd.DEVICE)
    try:
        classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=cd.DEVICE))
    except Exception as e: print(f"ERROR loading Classifier: {e}"); return
    classifier_model.eval()
    print("Classifier model loaded.")

    # --- Get Alpha Product Schedule ---
    alpha_prod_schedule = get_alpha_prod(
        time_steps=cd.TIME_STEPS, schedule_type=cd.SCHEDULE_TYPE,
        linear_start=cd.LINEAR_SCHEDULE_START, linear_end=cd.LINEAR_SCHEDULE_END,
        cosine_s=cd.COSINE_SCHEDULE_S, device=cd.DEVICE
    )

    # --- Load Data ---
    print("\nLoading MNIST test data...")
    _, test_loader = get_mnist_dataloaders(batch_size=num_images, root='./data')
    clean_input_images_batch, clean_input_labels_batch = next(iter(test_loader))
    clean_input_images = clean_input_images_batch.to(cd.DEVICE)
    clean_input_labels = clean_input_labels_batch # Kept on CPU or moved in accuracy fn
    print(f"Loaded {clean_input_images.shape[0]} clean test images with their labels.")

    # --- 0. Generate Adversarial Examples using FGSM ---
    print("\nStep 0: Generating FGSM adversarial examples...")
    adversarial_images = fgsm_attack(classifier_model, clean_input_images, clean_input_labels, fgsm_epsilon, cd.DEVICE)
    print(f"  Generated {adversarial_images.shape[0]} adversarial images.")
    
    # The input to the purification pipeline will be the adversarial images
    images_for_purification_input = adversarial_images

    # --- 1. VAE Encode adversarial images to get z_0_adv ---
    print("\nStep 1: Encoding adversarial images with VAE encoder...")
    with torch.no_grad():
        mu_sigma_map_adv = vae_encoder(images_for_purification_input)
        mu_adv = mu_sigma_map_adv[:, :cv.LATENT_DIM_CHANNELS, :, :]
        sigma_params_adv = mu_sigma_map_adv[:, cv.LATENT_DIM_CHANNELS:, :, :]
        z_0_adv = vae_model.reparameterize(mu_adv, sigma_params_adv)
    print(f"  Encoded adversarial latents (z_0_adv) shape: {z_0_adv.shape}")

    # --- 2. Get VAE direct reconstruction of adversarial images ---
    print("\nStep 2: Generating VAE direct reconstructions of adversarial images...")
    with torch.no_grad():
        reconstructed_images_vae_direct_from_adv = torch.clip(vae_decoder(z_0_adv), 0.0, 1.0)
    print(f"  VAE direct reconstructions of adv. images shape: {reconstructed_images_vae_direct_from_adv.shape}")

    # --- 3. Partial Forward Diffusion: Noise z_0_adv to z_N_PARTIAL_adv ---
    print(f"\nStep 3: Applying {n_partial_steps} analytical forward diffusion steps to z_0_adv...")
    t_target = n_partial_steps
    if not (1 <= t_target <= cd.TIME_STEPS):
        print(f"ERROR: n_partial_steps ({t_target}) invalid. Exiting."); return
    
    alpha_prod_t_target = alpha_prod_schedule[t_target - 1] 
    sqrt_alpha_prod_t_target = torch.sqrt(alpha_prod_t_target)
    sqrt_one_minus_alpha_prod_t_target = torch.sqrt(1.0 - alpha_prod_t_target)
    epsilon_noise_adv = torch.randn_like(z_0_adv, device=cd.DEVICE)
    with torch.no_grad():
        z_n_partial_adv = sqrt_alpha_prod_t_target * z_0_adv + sqrt_one_minus_alpha_prod_t_target * epsilon_noise_adv
    print(f"  Noised adversarial latents (z_{n_partial_steps}_adv) shape: {z_n_partial_adv.shape}")

    # --- 4. Partial Reverse Diffusion: Denoise z_N_PARTIAL_adv back to z_0_adv_reconstructed ---
    print(f"\nStep 4: Applying {n_partial_steps} reverse diffusion steps from z_{n_partial_steps}_adv...")
    current_latents_for_reverse_adv = z_n_partial_adv.clone() 
    for t_int_val in tqdm(range(n_partial_steps, 0, -1), desc="Partial reverse diffusion on adv latents"):
        t_tensor = torch.full((current_latents_for_reverse_adv.shape[0],), t_int_val, device=cd.DEVICE, dtype=torch.long)
        alpha_prod_t_val = alpha_prod_schedule[t_int_val - 1]
        alpha_prod_t_prev_val = alpha_prod_schedule[t_int_val - 2] if t_int_val > 1 else torch.tensor(1.0, device=cd.DEVICE)
        current_alpha_prod_reshaped = alpha_prod_t_val.view(-1, 1, 1, 1) 
        prev_alpha_prod_reshaped = alpha_prod_t_prev_val.view(-1, 1, 1, 1)
        with torch.no_grad():
            current_latents_for_reverse_adv = denoise_at_t(diffusion_model, current_latents_for_reverse_adv, t_tensor, current_alpha_prod_reshaped, prev_alpha_prod_reshaped, cd.DEVICE)
    z_0_reconstructed_via_diffusion_from_adv = current_latents_for_reverse_adv
    print(f"  Denoised adversarial latents (estimate of z_0_adv) shape: {z_0_reconstructed_via_diffusion_from_adv.shape}")

    # --- 5. VAE Decode the diffusion-reconstructed adversarial latents ---
    print("\nStep 5: Decoding the diffusion-reconstructed adversarial latents...")
    with torch.no_grad():
        reconstructed_images_diffusion_purified_adv = torch.clip(vae_decoder(z_0_reconstructed_via_diffusion_from_adv), 0.0, 1.0)
    print(f"  Purified adversarial images shape: {reconstructed_images_diffusion_purified_adv.shape}")

    # --- 6. Classifier Evaluation ---
    print("\nStep 6: Evaluating classifier accuracy...")
    acc_clean_standard = get_classification_accuracy(classifier_model, clean_input_images, clean_input_labels, cd.DEVICE)
    acc_adversarial_raw = get_classification_accuracy(classifier_model, adversarial_images, clean_input_labels, cd.DEVICE)
    acc_adv_vae_direct_recon = get_classification_accuracy(classifier_model, reconstructed_images_vae_direct_from_adv, clean_input_labels, cd.DEVICE)
    acc_adv_diffusion_purified_robust = get_classification_accuracy(classifier_model, reconstructed_images_diffusion_purified_adv, clean_input_labels, cd.DEVICE)

    print(f"\n--- Accuracy Results (on {num_images} images, FGSM eps={fgsm_epsilon}, T_partial={n_partial_steps}) ---")
    print(f"Standard Accuracy (on clean images):                         {acc_clean_standard*100:.2f}%")
    print(f"Accuracy on Adversarial Images (no defense):                 {acc_adversarial_raw*100:.2f}%")
    print(f"Accuracy on VAE Reconstructed Adversarial Images:            {acc_adv_vae_direct_recon*100:.2f}%")
    print(f"Robust Accuracy (Latent Diffusion Purified Adversarial):     {acc_adv_diffusion_purified_robust*100:.2f}%")

    # --- 7. Visualization ---
    print("\nStep 7: Plotting results...")
    clean_input_images_cpu = clean_input_images.cpu()
    adversarial_images_cpu = adversarial_images.cpu()
    reconstructed_images_diffusion_purified_adv_cpu = reconstructed_images_diffusion_purified_adv.cpu()

    fig, axes = plt.subplots(num_images, 3, figsize=(9, num_images * 3))
    if num_images == 1: axes = np.array([axes])

    title_str = (f"FGSM (eps={fgsm_epsilon}), Purify (T={n_partial_steps})\n"
                 f"Acc Clean: {acc_clean_standard*100:.1f}%, Acc Adv Raw: {acc_adversarial_raw*100:.1f}%\n"
                 f"Acc VAE Recon Adv: {acc_adv_vae_direct_recon*100:.1f}%, Acc Purified Adv: {acc_adv_diffusion_purified_robust*100:.1f}%")
    fig.suptitle(title_str, fontsize=10)
    
    for i in range(num_images):
        true_label = clean_input_labels[i].item()
        axes[i, 0].imshow(clean_input_images_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original Clean (L: {true_label})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(adversarial_images_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Adversarial (L: {true_label})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(reconstructed_images_diffusion_purified_adv_cpu[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title(f"Purified Adv (L: {true_label})")
        axes[i, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Adjust layout for suptitle
    
    os.makedirs(cd.FIGURES_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(cd.FIGURES_SAVE_DIR, f"adv_purify_acc_eps{fgsm_epsilon}_T{n_partial_steps}_N{num_images}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()
    print("\n--- Test Complete ---")

if __name__ == '__main__':
    NUM_IMAGES_TO_TEST = 4 
    N_PARTIAL_DIFF_STEPS = 100
    FGSM_EPSILON = 0.3 # Common epsilon values: 0.05, 0.1, 0.2, 0.3 for MNIST range [0,1]

    run_purification_and_evaluate_accuracy(
        num_images=NUM_IMAGES_TO_TEST, 
        n_partial_steps=N_PARTIAL_DIFF_STEPS,
        fgsm_epsilon=FGSM_EPSILON
    )
    
    # Example: Test with a different epsilon
    # FGSM_EPSILON_STRONG = 0.25
    # run_purification_and_evaluate_accuracy(
    #     num_images=NUM_IMAGES_TO_TEST, 
    #     n_partial_steps=N_PARTIAL_DIFF_STEPS,
    #     fgsm_epsilon=FGSM_EPSILON_STRONG
    # )

    # Example: Test with fewer diffusion steps
    # N_PARTIAL_DIFF_STEPS_FEW = 10
    # run_purification_and_evaluate_accuracy(
    #     num_images=NUM_IMAGES_TO_TEST, 
    #     n_partial_steps=N_PARTIAL_DIFF_STEPS_FEW,
    #     fgsm_epsilon=FGSM_EPSILON
    # )