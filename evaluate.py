# evaluate.py
import torch
import numpy as np

from dataset import get_processed_data
from model import Classifier

MODEL_PATH = 'mnist_classifier_weights.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Load Data ---
# We only need the validation set for evaluation
_, _, x_val, y_val = get_processed_data()

# --- Load Model ---
model = Classifier().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model weights loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_PATH}.")
    print("Please run train.py first to train and save the model.")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.eval() # Set model to evaluation mode

# --- Evaluation ---
print("\nStarting evaluation...")
all_preds = []
all_labels = []

with torch.no_grad(): # Disable gradient calculations
    # Process validation set
    # x_val is (N, H, W), add channel dim and send to device
    x_val_processed = x_val.unsqueeze(1).to(DEVICE) 
    y_val_labels = y_val # Keep labels on CPU for numpy comparison later

    outputs = model(x_val_processed) # Get logits
    
    # Get predicted classes (index of max logit)
    # Move predictions to CPU and convert to numpy for sklearn or direct numpy comparison
    predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy() 
    
    y_val_numpy = y_val_labels.numpy()


# Calculate accuracy
accuracy = np.mean(predicted_classes == y_val_numpy)
print(f'Model accuracy on validation set: {accuracy * 100.0:.2f}%')

# You can add more detailed evaluation, like a confusion matrix, here if needed.
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_val_numpy, predicted_classes, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val_numpy, predicted_classes))