import torch
import torch.nn as nn


class ModelMuscleJulius(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)

        # Calculate output size after convolutional layers
        conv_output_size = input_shape[1] - 4 - 4  # Input size reduced by kernel size - 1 twice

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * conv_output_size, 128)  # Adjusted size after convolutions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_shape[1])

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to (batch_size, channels, data_length) by adding channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer without activation (CrossEntropyLoss applies softmax)
        return x

    @staticmethod
    def to_string():
        return "muscle_julius"


class ModelMusclePero(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()

    def forward(self, x):
        pass

    @staticmethod
    def to_string():
        return "muscle_pero"


class ModelMuscleMarla(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()

    def forward(self, x):
        pass

    @staticmethod
    def to_string():
        return "muscle_marla"
