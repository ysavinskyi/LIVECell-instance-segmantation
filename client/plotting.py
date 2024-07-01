import matplotlib.pyplot as plt
import numpy as np


def plot_images(original_image, prediction_image):
    original_image_array = np.array(original_image)
    prediction_image_array = np.array(prediction_image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_image_array, cmap='gray')
    axes[0].set_title("Original Image (256x256)")
    axes[0].axis('off')

    # Display the prediction image
    axes[1].imshow(prediction_image_array, cmap='gray')
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    # Display the overlay of original and prediction
    axes[2].imshow(original_image_array, cmap='gray', interpolation='none')
    axes[2].imshow(prediction_image_array, cmap='jet', alpha=0.5, interpolation='none')  # Overlay with transparency
    axes[2].set_title("Overlay of Original and Prediction")
    axes[2].axis('off')

    plt.show()
