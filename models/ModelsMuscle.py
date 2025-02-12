import torch
import torch.nn as nn
import numpy as np


class ModelMuscle1(nn.Module):

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
        return "muscle_1"


class ModelMuscle2(nn.Module):

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
        return "muscle_2"


class ModelMuscle3(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Fully connected classifier
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_shape[1] // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape[1])
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    @staticmethod
    def to_string():
        return "muscle_3"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=427):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class ModelMuscle4(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        input_dim = 1
        model_dim = 64
        num_heads = 4
        num_layers = 3
        dropout = 0.1

        # Linear layer to map input (1D spectrum) into model_dim
        self.embedding = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Global Average Pooling
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Reduces sequence to single vector

        # Fully connected classification layer
        self.fc = nn.Linear(model_dim, output_shape[1])

    def forward(self, x):
        """
        x: Input of shape (batch_size, 427)
        """
        x = x.unsqueeze(-1)  # (batch_size, 427, 1)
        x = self.embedding(x)  # (batch_size, 427, model_dim)
        x = self.pos_encoder(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Required shape for Transformer: (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)  # Process with Transformer
        x = x.permute(1, 2, 0)  # Back to (batch_size, model_dim, seq_len)
        x = self.global_pooling(x).squeeze(-1)  # (batch_size, model_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

    @staticmethod
    def to_string():
        return "muscle_4"
