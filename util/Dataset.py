import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)


# TODO: read data and labels from file
class LargeDataset(Dataset):
    def __init__(self, data, labels, indices):
        self.input_shape = data.shape
        self.output_shape = labels.shape

        self.data = data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # map subset index to the actual index in the dataset
        actual_idx = self.indices[idx]
        # Load data and label as PyTorch tensors
        x = torch.tensor(self.data[actual_idx], dtype=torch.float32)
        y = torch.tensor(self.labels[actual_idx], dtype=torch.float32)
        return x, y

    def get_io(self):
        return self.input_shape, self.output_shape

    def create_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)


def get_train_test_indices(total_size, test_size=0.2):
    indices = list(range(total_size))
    indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=42)
    return indices_train, indices_test
