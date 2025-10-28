import numpy as np
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()

# Suppress scientific notation for easier reading
np.set_printoptions(suppress=True, linewidth=100)

print("--- Displaying first 5 samples from the MNIST Digits dataset ---")
print("Each 8x8 matrix represents a handwritten digit, with pixel intensity from 0-16.\n")

for i in range(5):
    print(f"Sample #{i+1}:")
    print(f"  Label: {digits.target[i]}")
    # The data is a flattened 64-pixel array. We reshape it to 8x8 to visualize.
    print(f"  8x8 Image Data:\n{digits.images[i]}")
    print("-"*30 + "\n")

