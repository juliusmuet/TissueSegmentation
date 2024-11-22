import torch
import torch.nn as nn


class ModelMuscleJulius1(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)

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
        return "muscle_julius_1"


class ModelMuscleJulius2(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(128)
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_shape[1])

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to (batch_size, channels, data_length) by adding channel dimension
        # Convolutional layers with ReLU and pooling
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def to_string():
        return "muscle_julius_2"


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
