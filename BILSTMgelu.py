from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Configuration
class Config:
    SR = 32000
    N_MELS = 128
    N_MFCC = 13
    MAX_SEQ_LEN = 200
    ROOT_FOLDER = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/'
    SUBSET_SIZE = 5000
    OUTPUT_FOLDER = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/preprocessed/'
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LR = 1e-4

CONFIG = Config()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, mfcc_files, mel_files, labels=None, transform=None):
        self.mfcc_files = mfcc_files
        self.mel_files = mel_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_image = Image.open(self.mel_files[idx]).convert('L')  # Convert to grayscale
        if self.transform:
            mel_image = self.transform(mel_image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            mfcc = torch.zeros((CONFIG.N_MFCC, CONFIG.MAX_SEQ_LEN))
            return mfcc, mel_image, label
        return mel_image

# Load train file paths and labels
def load_train_file_paths_and_labels(csv_path, subset_size=CONFIG.SUBSET_SIZE):
    df = pd.read_csv(csv_path)
    df_subset = df.sample(n=subset_size)
    file_paths = df_subset['path'].apply(lambda x: os.path.join(CONFIG.ROOT_FOLDER, x)).tolist()
    labels = df_subset['label'].tolist()
    return file_paths, labels

# Load test file paths
def load_test_file_paths(csv_path, subset_size=CONFIG.SUBSET_SIZE):
    df = pd.read_csv(csv_path)
    df_subset = df.sample(n=subset_size)
    file_paths = df_subset['path'].apply(lambda x: os.path.join(CONFIG.ROOT_FOLDER, x)).tolist()
    return file_paths

# Define the Bi-LSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 because it's bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(device)

        x = self.dropout(x)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(self.dropout(lstm_out[:, -1, :]))
        return x

# Define the CNN model for Mel-spectrogram images
class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))  # Replace F.relu with F.gelu
        x = self.pool(F.gelu(self.conv2(x)))  # Replace F.relu with F.gelu
        x = x.view(-1, 64 * 32 * 32)
        x = F.gelu(self.fc1(x))  # Replace F.relu with F.gelu
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Combine both models
class CombinedModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_output_dim, lstm_n_layers, lstm_bidirectional, lstm_dropout, cnn_output_dim):
        super(CombinedModel, self).__init__()
        self.lstm = BiLSTM(lstm_input_dim, lstm_hidden_dim, lstm_output_dim, lstm_n_layers, lstm_bidirectional, lstm_dropout)
        self.cnn = CNN(cnn_output_dim)
        self.fc = nn.Linear(lstm_output_dim + cnn_output_dim, 2)

    def forward(self, mfcc, mel):
        lstm_out = torch.zeros(mfcc.size(0), 128).to(device)  # Dummy LSTM output if no MFCC is provided
        if mfcc is not None:
            mfcc = mfcc.permute(0, 2, 1)  # Change from (batch, channels, seq_len) to (batch, seq_len, input_dim)
            lstm_out = self.lstm(mfcc)
        cnn_out = self.cnn(mel)
        combined = torch.cat((lstm_out, cnn_out), dim=1)
        out = self.fc(combined)
        return out

# Training and validation functions
def train(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for mfcc, mel, labels in loader:
        mfcc, mel, labels = mfcc.to(device), mel.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mfcc, mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    return epoch_loss / len(loader) if len(loader) > 0 else 0, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for mfcc, mel, labels in loader:
            mfcc, mel, labels = mfcc.to(device), mel.to(device), labels.to(device)
            outputs = model(mfcc, mel)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    return epoch_loss / len(loader) if len(loader) > 0 else 0, accuracy

# Main training loop
def main():
    # Load train data
    train_files, train_labels = load_train_file_paths_and_labels('/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train.csv')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])

    train_dataset = CustomDataset(train_files, train_files, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

    # Model initialization
    model = CombinedModel(
        lstm_input_dim=CONFIG.N_MFCC,
        lstm_hidden_dim=128,
        lstm_output_dim=128,
        lstm_n_layers=2,
        lstm_bidirectional=True,
        lstm_dropout=0.5,
        cnn_output_dim=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)

    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(CONFIG.N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        # Assuming you have validation data and loader (val_loader)
        # valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'best-model.pt')

        print(f'Epoch {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # Save the model weights in .h5 format
    model_path = 'model_weights.h5'
    torch.save(model.state_dict(), model_path)
    print(f'Model weights saved to {model_path}')

if __name__ == "__main__":
    main()