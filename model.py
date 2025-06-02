# model.py
import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = torch.nn.Sequential(
            # Input: B x 1 x 32 x 32
            torch.nn.Conv2d(1, 8, kernel_size=3, padding='same'), # Output: B x 8 x 32 x 32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),          # Output: B x 8 x 16 x 16
            
            torch.nn.Conv2d(8, 16, kernel_size=3, padding='same'), # Output: B x 16 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),          # Output: B x 16 x 8 x 8
            
            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),# Output: B x 32 x 8 x 8
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=4, stride=4),          # Output: B x 32 x 2 x 2
            
            torch.nn.Flatten(),                                   # Output: B x (32*2*2) = B x 128
            torch.nn.Dropout(p=0.5), # Dropout p=0.5 is common, article implies one is used
            
            torch.nn.Linear(32 * 2 * 2, 10),                      # Output: B x 10
        )
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, x):
        # x is expected to be (Batch, Channels, Height, Width)
        # e.g., (B, 1, 32, 32)
        y = self.layers(x)
        return y

if __name__ == '__main__':
    # Example usage:
    model = Classifier()
    print(f"Model architecture:\n{model}")
    print(f"Total trainable parameters: {model.num_params}") # Article states 16554
                                                             # My calculation for article's structure without explicit dropout p:
                                                             # Conv1: (1*3*3+1)*8 = 80
                                                             # Conv2: (8*3*3+1)*16 = 1168
                                                             # Conv3: (16*3*3+1)*32 = 4640
                                                             # Linear: (32*2*2 * 10) + 10 = 1280 + 10 = 1290
                                                             # Total: 80 + 1168 + 4640 + 1290 = 7178
                                                             # The article's count might include something specific or a slightly different interpretation.
                                                             # Let's re-check the article's structure: Linear(128, 10) means 128 input features
                                                             # After Flatten: 32 * 2 * 2 = 128 features. This matches.
                                                             # The parameter count of 16554 is a bit higher than my calculation.
                                                             # Let's re-calculate more carefully for the article code.
                                                             # Conv1: (1 * 3 * 3 + 1) * 8 = 72 + 8 = 80
                                                             # Conv2: (8 * 3 * 3 + 1) * 16 = 1152 + 16 = 1168
                                                             # Conv3: (16 * 3 * 3 + 1) * 32 = 4608 + 32 = 4640
                                                             # Linear: (128 * 10) + 10 = 1280 + 10 = 1290
                                                             # Total = 80 + 1168 + 4640 + 1290 = 7178.
                                                             # The article's parameter count of 16554 is significantly different.
                                                             # Perhaps the dropout layer is a `nn.Linear` layer in their count or a typo.
                                                             # My implementation matches the provided layers precisely.
                                                             # Let's assume my 7178 is correct for *these* layers.

    # Test with a dummy input
    dummy_input = torch.randn(4, 1, 32, 32) # Batch size 4, 1 channel, 32x32 image
    output = model(dummy_input)
    print(f"\nOutput shape for dummy input: {output.shape}") # Expected: torch.Size([4, 10])