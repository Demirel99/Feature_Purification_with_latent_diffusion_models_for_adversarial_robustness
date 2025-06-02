# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F # For cross_entropy
import math
import numpy as np

from dataset import get_processed_data
from model import Classifier

# --- Configuration ---
EPOCHS = 10 # Article uses 100, 10 is faster for testing
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LOG_INTERVAL = 1 # Log every epoch (article uses 10)
MODEL_SAVE_PATH = 'mnist_classifier_weights.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Load Data ---
x_train, y_train, x_val, y_val = get_processed_data()

# --- Initialize Model, Optimizer, Loss ---
model = Classifier().to(DEVICE)
print(f"Model initialized with {model.num_params} parameters.")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# The article uses cross_entropy directly, which is F.cross_entropy
# It expects raw logits from the model and class indices as targets.

# --- Training Loop ---
num_batches = math.ceil(x_train.shape[0] / BATCH_SIZE)
losses_history = [] # To store [train_loss, val_loss]

print(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train() # Set model to training mode (enables dropout)
    
    train_ids = torch.randperm(x_train.shape[0]) # Shuffle training data
    current_epoch_loss = 0.0

    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, x_train.shape[0])
        batch_ids = train_ids[start_idx:end_idx]

        # Data needs to be (N, C, H, W) for Conv2d
        # x_train is (N, H, W), so add channel dimension
        x_batch = x_train[batch_ids].unsqueeze(1).to(DEVICE) # Add channel dim, send to device
        y_batch = y_train[batch_ids].to(DEVICE) # Send to device

        # Forward pass
        y_pred_logits = model(x_batch)
        
        # Calculate loss
        loss = F.cross_entropy(y_pred_logits, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_epoch_loss += loss.item()

    average_epoch_loss = current_epoch_loss / num_batches

    # --- Validation (at the end of each epoch or LOG_INTERVAL) ---
    if epoch % LOG_INTERVAL == 0:
        model.eval() # Set model to evaluation mode (disables dropout)
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad(): # Disable gradient calculations for validation
            # Process validation set (can be done in one go if it fits memory)
            x_val_processed = x_val.unsqueeze(1).to(DEVICE) # Add channel dim, send to device
            y_val_labels = y_val.to(DEVICE)
            
            y_val_pred_logits = model(x_val_processed)
            val_loss = F.cross_entropy(y_val_pred_logits, y_val_labels, reduction='mean').item()
            
            # Calculate validation accuracy
            pred_classes = torch.argmax(y_val_pred_logits, dim=1)
            correct_val = (pred_classes == y_val_labels).sum().item()
            val_accuracy = 100.0 * correct_val / y_val.shape[0]

        losses_history.append([average_epoch_loss, val_loss])
        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {average_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')

# --- Save the trained model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nTraining complete. Model weights saved to {MODEL_SAVE_PATH}")

# --- Optional: Plot losses ---
import matplotlib.pyplot as plt
losses_np = np.array(losses_history)
plt.figure(figsize=(10, 5))
plt.plot(losses_np[:, 0], label='Training Loss')
plt.plot(losses_np[:, 1], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
print("Loss plot saved to loss_plot.png")
# plt.show() # Uncomment to display plot