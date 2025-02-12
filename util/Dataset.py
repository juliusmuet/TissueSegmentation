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

    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: list | np.ndarray):
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
        Returns the length of the used dataset, which corresponds to the number of indices.

        Returns:
            int: The number of samples in the subset of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
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

    def create_dataloader(self, batch_size: int, shuffle: bool = True):
        """
        Creates a PyTorch DataLoader for the dataset.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Shuffling of data (default: True)

        Returns:
            DataLoader: A DataLoader instance for the dataset with shuffling enabled.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def load_data_with_labels(base_path: str, shuffle: bool = True, seed: int | None = 42):
    """
    Loads .npy files from subdirectories under the base path, assigns one-hot encoded labels,
    and optionally shuffles the data.
    All data that belongs to one label must be located in the same subdirectory.
    The subdirectory's name is used for the string name of the label which is stored in the mapping dictionary.

    Args:
        base_path (str): Path to the base directory containing labeled subdirectories.
        shuffle (bool): Whether to shuffle the data and labels after loading.
        seed (int or None): Random seed for shuffling. If None, no seed is set.

    Returns:
        tuple: A tuple containing:
               - final_data (np.ndarray): Concatenated and optionally shuffled training data.
               - final_labels (np.ndarray): Corresponding one-hot encoded labels, shuffled if specified.
               - label_mapping (dict): Mapping of one-hot encoded labels to subdirectory names.

    Raises:
        ValueError: If no subdirectories found in base directory
    """
    data = []
    labels = []
    label_mapping = {}
    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    if not subdirs:
        raise ValueError(f"No subdirectories found in the base directory: {base_path}")

    for idx, subdir in enumerate(subdirs):
        one_hot_label = np.zeros(len(subdirs))
        one_hot_label[idx] = 1
        label_mapping[tuple(one_hot_label)] = subdir

        subdir_path = os.path.join(base_path, subdir)
        arrays = []

        for filename in os.listdir(subdir_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(subdir_path, filename)
                logging.info(f"Loading {file_path}")
                array = np.load(file_path)
                arrays.append(array)

        if arrays:
            concatenated_array = np.concatenate(arrays, axis=0)
            data.append(concatenated_array)
            # Repeat one-hot label for all samples in the subdirectory
            labels.append(np.tile(one_hot_label, (concatenated_array.shape[0], 1)))
        else:
            logging.warning(f"No .npy files found in the subdirectory: {subdir_path}")

    # Concatenate all data and labels
    final_data = np.concatenate(data, axis=0)
    final_labels = np.concatenate(labels, axis=0)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(final_data.shape[0])
        final_data = final_data[permutation]
        final_labels = final_labels[permutation]

    logging.info(f"Loaded {final_data.shape[0]} datapoints of dimension {final_data.shape[1]}")
    logging.info(f"Loaded {final_labels.shape[0]} labels of dimension {final_labels.shape[1]}")

    return final_data, final_labels, label_mapping


def decode_label_indices(label_indices: list | np.ndarray, label_mapping: dict):
    """
    Decodes label indices to their corresponding strings using a one-hot encoded dictionary.

    Args:
        label_indices (list or np.ndarray): A list of indices where each index represents the position
                                     in the one-hot encoded tuples where the value is 1.
        label_mapping (dict): A dictionary mapping one-hot encoded tuples (keys) to corresponding strings (values).

    Returns:
        list of str: A list of strings corresponding to the one-hot encoded tuples with 1s at the specified indices.

    Example:
        label_mapping = {
            (1, 0, 0): "First",
            (0, 1, 0): "Second",
            (0, 0, 1): "Third"
        }
        decode_label_indices([0, 2], label_mapping)  # Returns ["First", "Third"]
    """
    # List to store the results
    results = []

    # Iterate through each index in the list of label indices
    for label_index in label_indices:
        # Iterate through the dictionary to find the corresponding tuple
        for one_hot_tuple, string in label_mapping.items():
            if one_hot_tuple[label_index] == 1:  # Check if the value at the index is 1
                results.append(string)  # Add the corresponding string to results
                break  # Exit the loop once a match is found for this index

    return results


def get_one_hot_indices(one_hot_array: np.ndarray):
    """
    Given a one-hot encoded array, return the indices of ones.

    Parameters:
        one_hot_array (np.ndarray): A 2D numpy array where each row is a one-hot encoded vector.

    Returns:
        numpy.ndarray: An array of indices corresponding to the one-hot positions.
    """
    return np.argmax(one_hot_array, axis=1)


def get_train_test_indices(total_size: int, test_size: float = 0.2):
    """
    Splits a dataset into training and testing subsets by generating indices for each subset.

    Args:
        total_size (int): Total number of samples in the dataset.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
               - indices_train (list): List of indices for the training subset.
               - indices_test (list): List of indices for the testing subset.
    """
    indices = list(range(total_size))
    indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=42)
    return indices_train, indices_test
