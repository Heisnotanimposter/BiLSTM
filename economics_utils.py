import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class EconomicsDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x.unsqueeze(-1), y.unsqueeze(-1)

def prepare_economics_data(file_path, target_col, window_size=10):
    df = pd.read_csv(file_path)
    # Ensure date is sorted if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    values = df[target_col].values.reshape(-1, 1).astype(float)
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    return scaled_values.flatten(), scaler
