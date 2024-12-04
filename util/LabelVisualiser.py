import numpy as np
import matplotlib.pyplot as plt
from util.Dataset import decode_label_indices


def classify_image(model, image_array, label_to_string, output_filename):
    """
    Process the image data, predict labels for each pixel, and visualize/save the result.

    Parameters:
    - model (torch.nn.Module): PyTorch model (not used directly but passed to predict_function if required).
    - image_array (np.ndarray): 2D array where each pixel is represented by 427 values (H x W x 427).
    - label_to_string (dict): Dictionary for mapping of one-hot encode to string value.
    - output_filename (str): The name of the file where the visualized image will be saved.
    """
    # Get the original image dimensions
    height, width, depth = image_array.shape

    # Flatten the image (combine the height and width dimensions)
    flattened_pixels = image_array.reshape(-1, depth)  # Shape: (H*W, 427)

    # Predict labels for each pixel using the provided predict_function
    predicted_labels, _ = model.predict(flattened_pixels)  # Output: 1D array of length H*W

    # Reshape the predicted labels to match the original 2D image dimensions
    predicted_image = predicted_labels.reshape(height, width)

    # Dynamically generate colors based on the number of unique classes
    unique_labels = np.arange(0, len(label_to_string))
    num_classes = len(label_to_string)
    cmap = plt.cm.get_cmap('tab10', num_classes)  # Use tab10 colormap or any other suitable one
    label_to_color = {label: cmap(idx)[:3] for idx, label in enumerate(unique_labels)}

    # Convert colors to 0-255 range for visualization
    label_to_color = {label: (np.array(color) * 255).astype(np.uint8) for label, color in label_to_color.items()}

    # Map labels to colors
    color_image = np.zeros((*predicted_image.shape, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        color_image[predicted_image == label] = color

    # Save the visualized image using matplotlib
    plt.imshow(color_image)
    plt.axis('off')

    # Add a legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color) / 255, markersize=10, label=f"{decode_label_indices({label}, label_to_string)[0]}")
        for label, color in label_to_color.items()
    ]
    plt.legend(handles=legend_handles, loc='upper right', title="Classes")

    # Save the visualized image
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
