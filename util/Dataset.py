import logging
import torch.utils.data as data
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


class Dataset:

    """
    A class to handle dataset preprocessing and conversion for PyTorch models.

    This Dataset class performs standardization of input data,
    splits it into training and test sets, converts data to PyTorch tensors,
    and facilitates the creation of data loaders for training and evaluation.

    Attributes:
        input_shape (ndarray): The shape of the original input data.
        output_shape (ndarray): The shape of the labels.
        data (ndarray): Normalized input data after applying StandardScaler.
        labels (ndarray): Labels associated with the input data.
        device (torch.device): Device to use for tensors, either CPU or GPU.
        dataTensor (torch.Tensor): Tensor of normalized data.
        labelsTensor (torch.Tensor): Tensor of labels.
        X_trainTensor (torch.Tensor): Tensor of training data.
        X_testTensor (torch.Tensor): Tensor of test data.
        y_trainTensor (torch.Tensor): Tensor of training labels.
        y_testTensor (torch.Tensor): Tensor of test labels.
    """

    def __init__(self, _data, labels, test_train_ratio=0.2):
        """
        Initializes the Dataset object.

        Parameters:
            _data (ndarray): The input data to be processed.
            labels (ndarray): The target labels for the data.
            test_train_ratio (float): The ratio for splitting data into training and test sets. Default is 0.2.
        """
        self.input_shape = _data.shape
        self.output_shape = labels.shape

        # normalise data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(_data)
        self.labels = labels

        # split train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=test_train_ratio, random_state=42)

        # create tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataTensor = torch.Tensor(self.data).to(self.device)
        self.labelsTensor = torch.Tensor(self.labels).to(self.device)
        self.X_trainTensor = torch.Tensor(self.X_train).to(self.device)
        self.X_testTensor = torch.Tensor(self.X_test).to(self.device)
        self.y_trainTensor = torch.Tensor(self.y_train).to(self.device)
        self.y_testTensor = torch.Tensor(self.y_test).to(self.device)

    def __getitem__(self, _id):
        return self.dataTensor[_id], self.labelsTensor[_id]

    def __setitem__(self, _id):
        logging.warning("Setting items is not permitted in dataset, please create a new instance to change the data")
        return

    def get_io(self):
        return self.input_shape, self.output_shape

    def create_dataloader(self, batch_size):
        return data.DataLoader(data.TensorDataset(self.dataTensor, self.labelsTensor), shuffle=True, batch_size=batch_size)

    def create_dataloader_test_train(self, batch_size):
        loader_train = data.DataLoader(data.TensorDataset(self.X_trainTensor, self.y_trainTensor), shuffle=True, batch_size=batch_size)
        loader_test = data.DataLoader(data.TensorDataset(self.X_testTensor, self.y_testTensor), shuffle=True, batch_size=batch_size)

        return loader_train, loader_test
