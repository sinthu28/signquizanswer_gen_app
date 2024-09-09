import torch
import torch.nn as nn
import torch.optim as optim

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=100, input_size=(224, 224), num_channels=3):
        super(CNNLSTMModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        # Compute the size of the feature maps after the CNN layers
        conv_output_size = self._get_conv_output_size(input_size)
        
        self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def _get_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = self.cnn(x)
            return x.numel() // x.size(0)
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x