import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

"""
    NOT USED IN THIS PROJECT
    Part of future development of this framework to integrate Single Output Models into the ModelFactory
    without unnecessary code duplication.
    
    A class that wraps a machine learning model with ONLY ONE OUTPUT, managing training, evaluation,
    and prediction processes using PyTorch.
"""


class ModelSingleOutput:

    def __init__(self, model, dataset_train, dataset_test, input_shape, output_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(input_shape).to(self.device)
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
        if _input.shape == self.input_shape:
            return self.model(_input)
        else:
            logging.error("Input shape does not match expected input shape of model")
            return None

    def train(self, criterion, epochs, batch_size, save_dir='model_parameters'):
        criterion = nn.BCELoss()  #TODO: move criterion to factory

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

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"model_{self.model.to_string()}_{timestamp}.pth")
        os.makedirs(save_dir, exist_ok=True)    # Ensure the save directory exists
        torch.save(self.model.state_dict(), save_path)  # Save the model parameters
        logging.info(f"Model parameters saved to {save_path}")

    def evaluate(self):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.loader_test:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.model(X_batch)
                predicted = (y_pred > 0.5).float()

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        self.performance = 100 * correct / total
        print(f'Accuracy of the network {self.model.to_string()} on the test set: {self.performance:.2f}%')
        return self.performance

    def predict(self, data):
        self.model.eval()

        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predicted = self.model(data)

        return predicted.cpu().numpy()
