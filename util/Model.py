import os
from datetime import datetime
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from Dataset import Dataset
import logging

logging.basicConfig(level=logging.INFO)


class Model:

    """
    A class that wraps a machine learning model for MULTI-CLASS CLASSIFICATION (ONE-HOT), managing training, evaluation,
    and prediction processes using PyTorch.

    The Model class supports setting up an optimizer, handling the training loop,
    evaluating performance on a test set, and predicting new data.

    Attributes:
        loader_train (DataLoader): DataLoader for the training data.
        loader_test (DataLoader): DataLoader for the test data.
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output labels.
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        performance (float): Model accuracy on the test set.
    """

    def __init__(self, model, input_shape, output_shape):
        """
        Initializes the Model class with a specific neural network architecture.

        Parameters:
            model: The model class to instantiate.
            input_shape (tuple): The shape of the input data.
            output_shape (tuple): The shape of the output labels.
        """
        self.loader_train = None
        self.loader_test = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = model(input_shape, output_shape)
        self.optimizer = optim.Adam(self.model.parameters())
        self.performance = 0

    def __call__(self, _input):
        if _input.shape == self.input_shape:
            return self.model(_input)
        else:
            logging.error("Input shape does not match expected input shape of model")
            return None

    def train(self, dataset, criterion, epochs, batch_size, save_dir='model_parameters'):
        """
        Trains the model on the given dataset.

        Parameters:
            dataset (Dataset): Dataset object with training and test data loaders.
            criterion (torch.nn.Module): Loss function to use for training.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for the data loader.
            save_dir (str): Directory to save model parameters after training.

        Saves:
            Model parameters to a file named with a timestamp.
        """
        self.loader_train, self.loader_test = dataset.create_dataloader_test_train(batch_size=batch_size)

        if torch.cuda.is_available():
            self.model.cuda()
            logging.info("Using cuda library for training")
        else:
            logging.info("Cuda library is not available. Check your installation and hardware")

        self.model.train()

        for _ in tqdm(range(epochs), desc="Training: "):

            # training of the model: perform forward and backward pass with training data, then adjust weights
            for X_batch, y_batch in self.loader_train:

                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                self.optimizer.zero_grad()

                # forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)

                # backward pass
                loss.backward()
                self.optimizer.step()

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"model_{self.model.toString()}_{timestamp}.pth")
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Save the model parameters
        torch.save(self.model.state_dict(), save_path)
        logging.info(f"Model parameters saved to {save_path}")

    def evaluate(self):
        """
        Evaluates the model's performance on the test set.

        Returns:
            float: Accuracy of the model on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.loader_test:
                y_pred = self.model(X_batch)
                _, predicted = torch.max(y_pred.data, 1)
                _, actual = torch.max(y_batch, 1)
                total += y_batch.size(0)
                correct += (predicted == actual).sum().item()

        self.performance = 100 * correct / total
        logging.info(f'Accuracy of the network {self.model.toString()} on the test set: {self.performance:.2f}%')
        return self.performance

    def predict(self, data):
        """
        Makes predictions on new data.

        Parameters:
            data (array-like): New input data for predictions.

        Returns:
            ndarray: Predicted labels as a numpy array.
        """
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data_tensor = torch.Tensor(data)

        if torch.cuda.is_available():
            self.model.cuda()
            data_tensor = data_tensor.cuda()
            logging.info("Using cuda library for predicting")
        else:
            logging.info("Cuda library is not available. Check your installation and hardware")

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(data_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()
