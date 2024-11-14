import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

"""
    NOT USED IN THIS PROJECT
    
    A class that wraps a machine learning model with ONLY ONE OUTPUT, managing training, evaluation,
    and prediction processes using PyTorch.
"""

class ModelSingleOutput:

    def __init__(self, model, input_shape, output_shape):
        self.loader_train = None
        self.loader_test = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = model(input_shape)
        self.optimizer = optim.Adam(self.model.parameters())
        self.performance = 0

    def __call__(self, _input):
        if _input.shape == self.input_shape:
            return self.model(_input)
        else:
            logging.error("Input shape does not match expected input shape of model")
            return None

    def train(self, dataset, criterion, epochs, batch_size, save_dir='model_parameters'):
        criterion = nn.BCELoss() #TODO: move criterion to factory
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
                loss = criterion(y_pred.squeeze(), y_batch)

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
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.loader_test:
                y_pred = self.model(X_batch)
                predicted = (y_pred.squeeze() > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        self.performance = 100 * correct / total
        print(f'Accuracy of the network {self.model.toString()} on the test set: {self.performance:.2f}%')
        return self.performance

    def predict(self, data):
        data_tensor = torch.tensor(data)

        if torch.cuda.is_available():
            self.model.cuda()
            data_tensor.cuda()
            logging.info("Using cuda library for predicting")
        else:
            logging.info("Cuda library is not available. Check your installation and hardware")

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(data_tensor)

        return outputs.cpu().numpy()
