import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


class Model:
    """
    A wrapper class for managing a PyTorch model with MULTIPLE OUTPUT NEURONS, including training,
    evaluation, prediction, and saving/loading model parameters.

    Attributes:
        device (torch.device): The device on which the model is executed (CPU or GPU).
        model (torch.nn.Module): The PyTorch model instance.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output data (labels).
        dataset_train (Dataset): Training dataset object.
        dataset_test (Dataset): Testing dataset object.
        loader_train (DataLoader): DataLoader for the training dataset.
        loader_test (DataLoader): DataLoader for the testing dataset.
        performance (float): Model performance (accuracy) on the test set.
    """

    def __init__(self, model, dataset_train, dataset_test, input_shape, output_shape):
        """
        Initializes the Model class with the specified parameters.

        Args:
            model (class): The model class to instantiate.
            dataset_train (Dataset): The training dataset object.
            dataset_test (Dataset): The testing dataset object.
            input_shape (tuple): Shape of the input data.
            output_shape (tuple): Shape of the output data (labels).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(input_shape, output_shape).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

        logging.info(f"Using {self.device} for model operations")

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.loader_train = None
        self.loader_test = None

        self.performance = 0

    def __call__(self, _input):
        """
        Makes predictions using the model with the given input data.

        Args:
            _input (torch.Tensor): Input data for prediction.

        Returns:
            torch.Tensor or None: Model predictions if the input shape matches,
                                   otherwise None with an error logged.
        """
        if _input.shape == self.input_shape:
            return self.model(_input)
        else:
            logging.error("Input shape does not match expected input shape of model")
            return None

    def train(self, criterion, epochs, batch_size, save_dir='model_parameters'):
        """
        Trains the model using the specified loss function, number of epochs, and batch size.

        Args:
            criterion (nn.Module): The loss function for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for the DataLoader.
            save_dir (str, optional): Directory to save model parameters after training. Defaults to 'model_parameters'.
        """
        if self.loader_train is None or self.loader_test is None:
            self.loader_train = self.dataset_train.create_dataloader(batch_size)
            self.loader_test = self.dataset_test.create_dataloader(batch_size)

        self.model.train()

        for _ in tqdm(range(epochs), desc="Training: "):
            for X_batch, y_batch in self.loader_train:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

        # Create a timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"model_{self.model.to_string()}_{timestamp}.pth")
        os.makedirs(save_dir, exist_ok=True)    # Ensure the save directory exists
        torch.save(self.model.state_dict(), save_path)  # Save the model parameters
        logging.info(f"Model parameters saved to {save_path}")

    def evaluate(self):
        """
        Evaluates the model on the test dataset and calculates accuracy.

        Returns:
            float: The accuracy of the model on the test set.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.loader_test:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.model(X_batch)
                _, predicted = torch.max(y_pred.data, 1)
                _, actual = torch.max(y_batch, 1)

                total += y_batch.size(0)
                correct += (predicted == actual).sum().item()

        self.performance = 100 * correct / total
        logging.info(f'Accuracy of the network {self.model.to_string()} on the test set: {self.performance:.2f}%')
        return self.performance

    def predict(self, data):
        """
        Makes predictions on new data using the trained model.

        Args:
            data (np.ndarray or torch.Tensor): Input data for prediction.

        Returns:
            np.ndarray: Predicted class indices as a NumPy array.
        """
        self.model.eval()

        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()
