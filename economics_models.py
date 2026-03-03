import torch
import torch.nn as nn

class EconomicsBiLSTM(nn.Module):
    """
    BiLSTM model optimized for economic time-series forecasting.
    Includes Monte Carlo Dropout for variance estimation.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2, dropout=0.2):
        super(EconomicsBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Regression head
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the last hidden state from bidirectional sequence
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        last_hidden = lstm_out[:, -1, :]
        
        out = self.fc(self.dropout(last_hidden))
        return out

    def predict_with_variance(self, x, n_samples=50):
        """
        Predict with Monte Carlo dropout to estimate variance.
        """
        self.train() # Enable dropout during inference
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(x).unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0) # (n_samples, batch_size, output_dim)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        return mean, variance
