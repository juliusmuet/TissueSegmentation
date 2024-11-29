import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class LargeDataset(Dataset):
    """
    A PyTorch Dataset class for handling large datasets, where only a subset of
    data is accessed using specified indices. It provides methods for creating
    data loaders and accessing input/output shapes.

    Attributes:
        input_shape (tuple): Shape of the input data array.
        output_shape (tuple): Shape of the labels array.
        data (np.ndarray): The dataset containing input data.
        labels (np.ndarray): The dataset containing corresponding labels.
        indices (list or np.ndarray): List of indices specifying the subset of data to be used.
    """

    def __init__(self, data, labels, indices):
        """
        Initializes the dataset with data, labels, and subset indices.

        Args:
            data (np.ndarray): Numpy array containing the input data.
            labels (np.ndarray): Numpy array containing the corresponding labels.
            indices (list or np.ndarray): List or array of indices specifying
                                           the subset of data to include in this dataset.
        """
        self.input_shape = data.shape
        self.output_shape = labels.shape
        self.data = data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        """
        Returns the length of the dataset, which corresponds to the number of indices.

        Returns:
            int: The number of samples in the subset of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample and its corresponding label.

        Args:
            idx (int): The index of the sample in the subset.

        Returns:
            tuple: A tuple (x, y), where:
                   - x (torch.Tensor): Input data sample as a PyTorch tensor.
                   - y (torch.Tensor): Corresponding label as a PyTorch tensor.
        """
        # map subset index to the actual index in the dataset
        actual_idx = self.indices[idx]
        # Load data and label as PyTorch tensors
        x = torch.tensor(self.data[actual_idx], dtype=torch.float32)
        y = torch.tensor(self.labels[actual_idx], dtype=torch.float32)
        return x, y

    def get_io(self):
        """
        Retrieves the input and output shapes of the dataset.

        Returns:
            tuple: A tuple (input_shape, output_shape), where:
                   - input_shape (tuple): Shape of the input data.
                   - output_shape (tuple): Shape of the labels.
        """
        return self.input_shape, self.output_shape

    def create_dataloader(self, batch_size):
        """
        Creates a PyTorch DataLoader for the dataset.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: A DataLoader instance for the dataset with shuffling enabled.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=True)


def load_and_label_npy(directory, is_muscle):
    """
    Loads .npy files from the given directory, concatenates them along the first dimension,
    and generates one-hot encoded labels.

    Args:
        directory (str): Path to the directory containing .npy files.
        is_muscle (bool): If True, generates labels [1, 0] (muscle).
                          If False, generates labels [0, 1] (not a muscle).

    Returns:
        tuple: A tuple containing:
               - concatenated_array (np.ndarray): The concatenated numpy array.
               - labels (np.ndarray): One-hot encoded labels for the data.
    """
    # Initialize an empty list to store arrays
    arrays = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  # Process only .npy files
            file_path = os.path.join(directory, filename)
            print(f"Loading {file_path}")
            # Load the .npy file and append to the list
            array = np.load(file_path)
            arrays.append(array)

    if not arrays:
        raise ValueError(f"No .npy files found in the directory: {directory}")

    # Concatenate all arrays along the first dimension
    concatenated_array = np.concatenate(arrays, axis=0)

    # Generate one-hot encoded labels
    label_value = [1, 0] if is_muscle else [0, 1]
    labels = np.tile(label_value, (concatenated_array.shape[0], 1))

    return concatenated_array, labels


def load_and_merge_two_directories(dir_muscle, dir_other, shuffle=True, seed=42):
    """
    Loads data from two directories, assigns one-hot encoded labels (muscle=[1,0], not muscle=[0,1]),
    concatenates both data and labels, and shuffles them if specified.

    Args:
        dir_muscle (str): Path to the first directory (muscle).
        dir_other (str): Path to the second directory (not muscle).
        shuffle (bool): Whether to shuffle the data and labels after concatenation.
        seed (int or None): Random seed for shuffling. If None, no seed is set.

    Returns:
        tuple: A tuple containing:
               - final_data (np.ndarray): Concatenated and optionally shuffled training data.
               - final_labels (np.ndarray): Corresponding one-hot encoded labels, shuffled if specified.
    """
    # Load and label data from the first directory (muscle)
    data_one, labels_one = load_and_label_npy(dir_muscle, is_muscle=True)

    # Load and label data from the second directory (not muscle)
    data_two, labels_two = load_and_label_npy(dir_other, is_muscle=False)

    # Concatenate data and labels from both directories
    merged_data = np.concatenate([data_one, data_two], axis=0)
    merged_labels = np.concatenate([labels_one, labels_two], axis=0)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        # Create a permutation of indices and apply it to both data and labels
        permutation = np.random.permutation(merged_data.shape[0])
        merged_data = merged_data[permutation]
        merged_labels = merged_labels[permutation]

    return merged_data, merged_labels


def get_train_test_indices(total_size, test_size=0.2):
    """
    Splits a dataset into training and testing subsets by generating indices for each subset.

    Args:
        total_size (int): Total number of samples in the dataset.
        test_size (float or int, optional): Proportion of the dataset to include in the test split
                                            (if float, between 0.0 and 1.0), or the absolute number
                                            of test samples (if int). Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
               - indices_train (list): List of indices for the training subset.
               - indices_test (list): List of indices for the testing subset.
    """
    indices = list(range(total_size))
    indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=42)
    return indices_train, indices_test
