import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from util.Dataset import LargeDataset, get_one_hot_indices
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

    def __init__(self, model, dataset_train: LargeDataset, dataset_test: LargeDataset, input_shape: tuple, output_shape: tuple):
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

    def __call__(self, _input: np.ndarray | torch.Tensor):
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

    def _calculate_accuracy(self, dataloader: DataLoader):
        """
        Helper function to calculate accuracy on a given dataloader.

        Args:
            dataloader (DataLoader): The DataLoader to evaluate.

        Returns:
            float: Accuracy percentage.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.model(X_batch)
                _, predicted = torch.max(y_pred.data, 1)
                _, actual = torch.max(y_batch, 1)

                total += y_batch.size(0)
                correct += (predicted == actual).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def train(self, criterion, epochs: int, batch_size: int, save_dir: str = 'model_parameters', evaluate_during_training: bool = True):
        """
        Trains the model using the specified loss function, number of epochs, and batch size.

        Args:
            criterion (nn.Module): The loss function for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for the DataLoader.
            save_dir (str, optional): Directory to save model parameters after training. Defaults to 'model_parameters'.
            evaluate_during_training (bool, optional): If True, evaluates the model on train and test datasets after each epoch. Defaults to True.
        """
        if self.loader_train is None or self.loader_test is None:
            self.loader_train = self.dataset_train.create_dataloader(batch_size)
            self.loader_test = self.dataset_test.create_dataloader(batch_size)

        self.model.train()

        for epoch in tqdm(range(epochs), desc="Training: "):
            for X_batch, y_batch in self.loader_train:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            # Evaluate the model if parameter is set
            if evaluate_during_training:
                train_accuracy = self._calculate_accuracy(self.loader_train)
                test_accuracy = self._calculate_accuracy(self.loader_test)
                logging.info(f"Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

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
        if self.loader_train is None or self.loader_test is None:
            logging.info(f"Train the model before evaluation")
            return 0.0

        self.performance = self._calculate_accuracy(self.loader_test)
        logging.info(f'Accuracy of the network {self.model.to_string()} on the test set: {self.performance:.2f}%')
        return self.performance

    def predict(self, data: np.ndarray | torch.Tensor, batch_size: int = 1024):
        """
        Makes predictions on new data using the trained model and calculates probabilities.

        Args:
            data (np.ndarray or torch.Tensor): Input data for prediction.
            batch_size (int): Number of inputs to process in a single batch.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Predicted class indices as a NumPy array.
                - np.ndarray: Probabilities for each class as a NumPy array.
        """
        self.model.eval()
        num_pixels = data.shape[0]
        all_predictions = []
        all_probabilities = []

        # Ensure input data is a PyTorch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            for start_idx in range(0, num_pixels, batch_size):
                # Slice the batch
                batch = data[start_idx:start_idx + batch_size].to(self.device)

                # Predict on the batch
                outputs = self.model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)

                # Collect results
                all_predictions.append(predicted.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)

        return all_predictions, all_probabilities

    def calculate_statistics(self, spectra: np.ndarray, labels: np.ndarray, mapping: dict, output_dir: str):
        """
        Computes and saves classification statistics for a given model, including a confusion matrix and classification report.

        Parameters:
            spectra (np.ndarray): Input features for making predictions.
            labels (np.ndarray): True labels corresponding to the input spectra in one-hot-encoded format.
            mapping (dict): A dictionary mapping label indices to class names.
            output_dir (str): Directory where the output files will be saved.

        Outputs:
            - Saves the confusion matrix as an image file.
            - Saves the classification report as a text file.
        """
        predictions, _ = self.predict(spectra)
        labels = get_one_hot_indices(labels)

        output_path_cm = os.path.join(output_dir, f"{self.model.to_string()}_confusion_matrix.png")
        cm = confusion_matrix(labels, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping.values())
        cm_display.plot()
        cm_display.figure_.savefig(output_path_cm, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion Matrix of {self.model.to_string()} saved under {output_path_cm}")

        output_path_report = os.path.join(output_dir, f"{self.model.to_string()}_classification_report.txt")
        report = classification_report(labels, predictions, target_names=mapping.values())
        with open(output_path_report, 'w') as file:
            file.write(report)
        logging.info(f"Classification Report of {self.model.to_string()} saved under {output_path_report}")
