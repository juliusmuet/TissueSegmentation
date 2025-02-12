import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from util.Model import Model
from util.Dataset import decode_label_indices


def load_and_segment_images(base_path: str, model: Model, mapping: dict, output_path: str = "data/images_results"):
    """
    Loads .npy files, representing images with spectrum information per pixel,
    from a directory, classifies their content using a model, and saves the classification results in an image.

    Args:
        base_path (str): The directory path containing .npy files to process.
        model (Model): The PyTorch model instance.
        mapping (dict): Dictionary for mapping of one-hot encode to string value.
        output_path (str): The directory path for saving the segmented images.

    Raises:
        ValueError: If the provided `base_path` is not a valid directory.
    """
    # Ensure the directory exists
    if not os.path.isdir(base_path):
        raise ValueError(f"Provided path '{base_path}' is not a valid directory.")

    # List all .npy files in the directory
    npy_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]

    for npy_file in npy_files:
        # Get the file name without extension
        file_name = os.path.splitext(npy_file)[0]

        # Load the numpy array
        file_path = os.path.join(base_path, npy_file)
        image_array = np.load(file_path)

        # Classify each image pixel / spectrum
        logging.info(f"Classifying {output_path}/output_{model.model.to_string()}_{file_name}")
        predicted_image = predict_image(image_array, model)
        segment_image(predicted_image, mapping, f"{output_path}/output_{model.model.to_string()}_{file_name}")


def predict_image(image_array: np.ndarray, model: Model):
    """
    Predicts the spectra of an image based on the input image array and the provided model.

    This function takes an input image array and a pre-trained model, performs pixel-wise predictions,
    and returns the resulting 2D array of predicted labels that match the original image dimensions.

    Parameters:
    image_array (numpy.ndarray):
        A 3D array representing the input image, with dimensions (height, width, depth).
        - height: The number of rows (pixels) in the image.
        - width: The number of columns (pixels) in the image.
        - depth: The number of channels per pixel (e.g., 427 for FT-IR data).

    model (Model): A pre-trained model for prediction

    Returns:
    numpy.ndarray: A 2D array of predicted labels with dimensions (height, width), where each element corresponds
        to the predicted label for the respective pixel in the input image.
    """
    # Get the original image dimensions
    height, width, depth = image_array.shape

    # Flatten the image (combine the height and width dimensions)
    flattened_pixels = image_array.reshape(-1, depth)  # Shape: (H*W, depth)

    # Predict labels for each pixel using the provided model
    predicted_labels, _ = model.predict(flattened_pixels)  # Output: 1D array of length H*W

    # Reshape the predicted labels to match the original 2D image dimensions
    return predicted_labels.reshape(height, width)


def segment_image(image_labels_array: np.ndarray, label_to_string: dict, output_filename: str):
    """
    Visualize the classifications of an image and save the segmented image.

    Parameters:
    - image_labels_array (np.ndarray): 2D array where each pixel represents its label (H x W).
    - label_to_string (dict): Dictionary for mapping of one-hot encode to string value.
    - output_filename (str): The name of the file where the visualized image will be saved.
    """
    # Dynamically generate colors based on the number of unique classes
    unique_labels = np.arange(0, len(label_to_string))
    num_classes = len(label_to_string)
    cmap = plt.cm.get_cmap('tab10', num_classes)  # Use tab10 colormap or any other suitable one
    label_to_color = {label: cmap(idx)[:3] for idx, label in enumerate(unique_labels)}

    # Convert colors to 0-255 range for visualization
    label_to_color = {label: (np.array(color) * 255).astype(np.uint8) for label, color in label_to_color.items()}

    # Map labels to colors
    color_image = np.zeros((*image_labels_array.shape, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        color_image[image_labels_array == label] = color

    # Save the visualized image using matplotlib
    plt.imshow(color_image)
    plt.axis('off')

    # Add a legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color) / 255, markersize=10, label=f"{decode_label_indices({label}, label_to_string)[0]}")
        for label, color in label_to_color.items()
    ]
    plt.legend(
        handles=legend_handles,
        loc='upper right',
        title="Classes",
        title_fontsize=8,  # Reduce the title font size
        fontsize=6,        # Reduce the label font size
        markerscale=0.7    # Scale down the size of the markers in the legend
    )

    # Save the visualized image
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
