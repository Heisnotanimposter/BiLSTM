import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from economics_models import EconomicsBiLSTM
from economics_utils import prepare_economics_data, EconomicsDataset
import os

def train_economics():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 5
    batch_size = 4
    epochs = 100
    
    # Try multiple files to demonstrate diversity
    files = [
        ('Stock_Data.csv', 'Close'),
        ('Inflation_Rate_Data.csv', 'Inflation_Rate'),
        ('GDP_Data.csv', 'GDP_Value')
    ]
    
    # For demonstration, we'll train on Stock Data
    file_path, target_col = files[0]
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    data, scaler = prepare_economics_data(file_path, target_col, window_size)
    dataset = EconomicsDataset(data, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EconomicsBiLSTM(input_dim=1, hidden_dim=32, n_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training Economics BiLSTM on {file_path}...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.6f}")
            
    # Save model
    torch.save(model.state_dict(), 'economics_model.pt')
    print("Model saved to economics_model.pt")

if __name__ == "__main__":
    train_economics()
