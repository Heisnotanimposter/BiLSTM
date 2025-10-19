"""
Model definitions for BiLSTM Audio Classification Project
Includes BiLSTM, CNN, and combined models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM model for sequence classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3):
        """
        Initialize BiLSTM model
        
        Args:
            input_dim: Input feature dimension (e.g., number of MFCC coefficients)
            hidden_dim: Hidden dimension of LSTM
            output_dim: Output dimension
            n_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
        """
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Calculate output dimension based on bidirectional setting
        num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * num_directions
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden states
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.n_layers * num_directions, 
            x.size(0), 
            self.hidden_dim
        ).to(x.device)
        c_0 = torch.zeros(
            self.n_layers * num_directions, 
            x.size(0), 
            self.hidden_dim
        ).to(x.device)
        
        # Apply dropout
        x = self.dropout(x)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Use the last hidden state
        # Concatenate hidden states from all layers
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_n_combined = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        else:
            h_n_combined = h_n[-1, :, :]
        
        # Apply dropout and fully connected layer
        output = self.fc(self.dropout(h_n_combined))
        
        return output


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification
    Used for processing Mel-spectrogram images
    """
    
    def __init__(self, output_dim: int, activation: str = 'relu'):
        """
        Initialize CNN model
        
        Args:
            output_dim: Output dimension
            activation: Activation function ('relu', 'gelu', or 'leaky_relu')
        """
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        # Assuming input image size of 128x128 after 3 pooling operations: 128/8 = 16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First conv block
        x = self.pool(self.activation(self.conv1(x)))
        
        # Second conv block
        x = self.pool(self.activation(self.conv2(x)))
        
        # Third conv block
        x = self.pool(self.activation(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CombinedModel(nn.Module):
    """
    Combined CNN-BiLSTM model for audio classification
    Uses CNN for Mel-spectrogram images and BiLSTM for MFCC features
    """
    
    def __init__(self, lstm_input_dim: int, lstm_hidden_dim: int, 
                 lstm_output_dim: int, lstm_n_layers: int, 
                 lstm_bidirectional: bool, lstm_dropout: float,
                 cnn_output_dim: int, cnn_activation: str = 'relu'):
        """
        Initialize Combined model
        
        Args:
            lstm_input_dim: Input dimension for LSTM
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_output_dim: Output dimension for LSTM
            lstm_n_layers: Number of LSTM layers
            lstm_bidirectional: Whether to use bidirectional LSTM
            lstm_dropout: Dropout rate for LSTM
            cnn_output_dim: Output dimension for CNN
            cnn_activation: Activation function for CNN
        """
        super(CombinedModel, self).__init__()
        
        # Initialize sub-models
        self.lstm = BiLSTM(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=lstm_output_dim,
            n_layers=lstm_n_layers,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout
        )
        
        self.cnn = CNN(
            output_dim=cnn_output_dim,
            activation=cnn_activation
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(lstm_output_dim + cnn_output_dim, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, mfcc, mel):
        """
        Forward pass
        
        Args:
            mfcc: MFCC features of shape (batch_size, n_mfcc, seq_len)
            mel: Mel-spectrogram image of shape (batch_size, 1, height, width)
        
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        # Process MFCC with BiLSTM
        # Transpose to (batch_size, seq_len, n_mfcc)
        mfcc = mfcc.permute(0, 2, 1)
        lstm_out = self.lstm(mfcc)
        
        # Process Mel-spectrogram with CNN
        cnn_out = self.cnn(mel)
        
        # Concatenate outputs
        combined = torch.cat((lstm_out, cnn_out), dim=1)
        
        # Final classification
        output = self.fc(self.dropout(combined))
        
        return output


class AudioClassifier(nn.Module):
    """
    Simplified audio classifier using only BiLSTM on MFCC features
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3):
        """
        Initialize Audio Classifier
        
        Args:
            input_dim: Input feature dimension (number of MFCC coefficients)
            hidden_dim: Hidden dimension of LSTM
            n_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
        """
        super(AudioClassifier, self).__init__()
        
        self.model = BiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,  # Binary classification
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        # Transpose to (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        return self.model(x)

