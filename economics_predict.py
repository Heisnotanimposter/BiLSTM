import torch
import numpy as np
from economics_models import EconomicsBiLSTM
from economics_utils import prepare_economics_data
import os

def predict_economics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 5
    file_path = 'Stock_Data.csv'
    target_col = 'Close'
    
    if not os.path.exists('economics_model.pt'):
        print("Pre-requisite: Please run economics_train.py first.")
        return

    data, scaler = prepare_economics_data(file_path, target_col, window_size)
    
    model = EconomicsBiLSTM(input_dim=1, hidden_dim=32, n_layers=2).to(device)
    model.load_state_dict(torch.load('economics_model.pt', map_location=device))
    
    # Take the last window for prediction
    last_window = data[-window_size:]
    x = torch.FloatTensor(last_window).view(1, window_size, 1).to(device)
    
    # Predict with variance (Total Variance)
    mean, variance = model.predict_with_variance(x, n_samples=100)
    
    # Invert scaling
    predicted_val = scaler.inverse_transform(mean.cpu().numpy())[0][0]
    total_variance = variance.cpu().numpy()[0][0] # Variance in scaled domain
    
    print("\n" + "="*40)
    print("ECONOMICS FORECAST RESULTS")
    print("="*40)
    print(f"Input Sequence (Scaled): {last_window}")
    print(f"Predicted Next Value: {predicted_val:.2f}")
    print(f"Total Prediction Variance: {total_variance:.6f}")
    print("="*40)

if __name__ == "__main__":
    predict_economics()
