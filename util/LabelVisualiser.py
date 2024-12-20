import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import getpass
import smtplib
from email.message import EmailMessage
from util.Dataset import decode_label_indices


def load_and_classify_images(base_path, model, mapping):
    """
    Loads .npy files from a directory, classifies their content using a model,
    and saves the classification results in an image.

    Args:
        base_path (str): The directory path containing .npy files to process.
        model (Model): The PyTorch model instance.
        mapping (dict): Dictionary for mapping of one-hot encode to string value.

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
        array = np.load(file_path)

        # Apply the provided function
        classify_image(model, array, mapping, f"{base_path}/results/output_{model.model.to_string()}_{file_name}")


def classify_image(model, image_array, label_to_string, output_filename):
    """
    Process the image data, predict labels for each pixel, and visualize/save the result.

    Parameters:
    - model (torch.nn.Module): PyTorch model (not used directly but passed to predict_function if required).
    - image_array (np.ndarray): 2D array where each pixel is represented by 427 values (H x W x 427).
    - label_to_string (dict): Dictionary for mapping of one-hot encode to string value.
    - output_filename (str): The name of the file where the visualized image will be saved.
    """
    logging.info(f"Classifying {output_filename}")

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

    # Send image via email
    send_classified_image(output_filename)


def send_classified_image(file_path):
    """
    Sends an email over web.de with a classified image attached.

    Parameters:
    - file_path (str): The path to the image file to be attached to the email.
    - recipient_email (str): The email address of the recipient.
    - sender_email (str): The email address of the sender.
    - sender_password (str): The password for the sender's email account.

    The function creates an email message, attaches the specified image file, and sends the email
    using an SMTP server (configured for Gmail in this example). Logging is used to track success or failure.

    Raises:
    - Logs an error message if the email fails to send due to any exception.
    """
    file_path = file_path + ".png"
    try:
        # Create the email message
        msg = EmailMessage()
        msg['Subject'] = f"Classified Image: {file_path}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content(f"Please find attached the classified image: {file_path}.")

        # Attach the image file
        with open(file_path, 'rb') as img_file:
            img_data = img_file.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=file_path)

        # Send the email
        with smtplib.SMTP_SSL('smtp.web.de', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        logging.info(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


# Get email data from keyboard input
recipient_email = input("Enter the recipient's email address: ")
sender_email = input("Enter the sender's email address: ")
sender_password = getpass.getpass("Enter the sender's email password: ")
