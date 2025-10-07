import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_mask(mask_array, title="Mask Visualization"):
    """Display a multi-class segmentation mask."""
    plt.figure(figsize=(6,6))
    plt.imshow(mask_array, cmap="nipy_spectral")
    plt.title(title)
    plt.axis("off")
    plt.show()

def save_mask(mask_array, path):
    Image.fromarray(mask_array.astype(np.uint8)).save(path)     #Save mask array as PNG.

def load_mask(path):
    return np.array(Image.open(path).convert("L")) #Load a grayscale mask image and return as numpy array.

def create_binary_mask(mask_array, label_id):
    return np.where(mask_array == label_id, 255, 0).astype(np.uint8) #Return binary mask (255 where mask == label_id).
