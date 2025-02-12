import torch.nn as nn
import torch

"""
    NOT USED IN THIS PROJECT
    Part of future development of this framework to integrate Single Output Models into the ModelFactory
    without unnecessary code duplication.
    
    An example neural network with only one output
"""


class ModelTestSingleOutput(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    @staticmethod
    def to_string():
        return "test_single_output"
