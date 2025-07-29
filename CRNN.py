import torch.nn as nn
import torch.nn.functional as nnF

# input_height is the number of frequency bins in the MFCC

class CRNN(nn.Module):
    def __init__(self, input_height, input_channels=1, conv_channels=16, hidden_size=128, num_classes=2):
        super(CRNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels * 2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.05)

        # Calculate output dimensions after convolution and pooling
        conv_output_height = input_height // 4  # Two pooling layers reduce height

        # Calculate LSTM input size
        self.lstm_input_size = (conv_channels * 2) * conv_output_height

        # Initialize LSTM and Fully Connected Layers
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features=num_classes)

    def forward(self, x):
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnF.relu(x)
        x = self.pool(x)
        # x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nnF.relu(x)
        x = self.pool(x)
        # x = self.dropout(x)

        # Reshape for LSTM      
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, freq)
        batch_size, time_steps, channels, freq = x.shape
        
        # Calculate lstm_input_size based on observed shapes
        x = x.contiguous().view(batch_size, time_steps, -1)

        # LSTM
        x, _ = self.lstm(x)

        # Classification
        x = x[:, -1, :]
        x = self.fc(x)
        return x