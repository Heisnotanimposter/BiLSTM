# BiLSTM Audio Classification Project

A comprehensive deep learning project for audio classification using Bidirectional Long Short-Term Memory (BiLSTM) networks. This project focuses on distinguishing between real and fake audio samples using MFCC (Mel-Frequency Cepstral Coefficients) features and Mel-spectrogram images.

## 🎯 Project Overview

This project implements a robust audio classification system that can:
- Extract MFCC features from audio files
- Generate Mel-spectrogram images
- Train BiLSTM and CNN-BiLSTM hybrid models for audio classification
- **[NEW] Economics Time-Series Forecasting**: BiLSTM architecture optimized for financial data
- **[NEW] Total Variance Analysis**: Monte Carlo Dropout for prediction uncertainty estimation
- **[NEW] 3D Neural Flow Visualizer**: Real-time Three.js landing page for architecture visualization
- Evaluate model performance with comprehensive metrics
- Generate predictions for new audio samples

## 📁 Project Structure

```
BiLSTM/
├── config.py                 # Centralized configuration management
├── models.py                 # Model definitions (BiLSTM, CNN, Combined)
├── utils.py                  # Utility functions (data loading, preprocessing, evaluation)
├── train.py                  # Main training script
├── predict.py                # Inference script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # License information
│
├── data/                     # Data directory (create this)
│   └── TeamDeepwave/
│       └── dataset/
│           └── open/
│               ├── train/    # Training audio files (.ogg)
│               ├── train.csv # Training labels
│               └── test.csv  # Test data
│
├── models/                   # Saved models (auto-created)
├── outputs/                  # Output files (auto-created)
└── logs/                     # Training logs (auto-created)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd BiLSTM

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Before running the training script, update the paths in `config.py` to match your local setup:

```python
# In config.py
DATASET_ROOT = DATA_DIR / "TeamDeepwave" / "dataset" / "open"
TRAIN_DIR = DATASET_ROOT / "train"
TRAIN_CSV = DATASET_ROOT / "train.csv"
```

### 3. Data Preparation

Place your audio files in the appropriate directory structure:
- Training audio files: `data/TeamDeepwave/dataset/open/train/`
- Training CSV with labels: `data/TeamDeepwave/dataset/open/train.csv`

The CSV file should have the following format:
```csv
path,label
train/audio1.ogg,real
train/audio2.ogg,fake
...
```

### 4. Training

#### Basic Training (extract features on-the-fly):
```bash
python train.py
```

#### Training with Preprocessed Features:
```bash
# First run extracts and saves features
python train.py

# Subsequent runs can use preprocessed features
python train.py --use-preprocessed
```

### 5. Inference

### 6. Economics Forecasting
    ```bash
    # Train the economics model
    conda run -n base python economics_train.py
    
    # Run prediction with Total Variance
    conda run -n base python economics_predict.py
    ```

### 7. 3D Visualization
    Open `web/index.html` in your favorite browser to view the real-time neural flow visualizer.

## 📈 Economics Engine
The project now includes a dedicated engine for economics forecasting:
- **`economics_models.py`**: BiLSTM optimized for 1D time-series data.
- **`economics_utils.py`**: Sliding window preprocessing and data scaling.
- **`economics_train.py`**: Training pipeline for diversified datasets (Stock, GDP, Inflation).
- **`economics_predict.py`**: Inference with Monte Carlo Dropout-based **Total Variance**.

## 🎨 3D Visualization Landing Page
A premium landing page built with Three.js provides:
- **Interactive Neural Map**: Visualizes data flow through BiLSTM layers.
- **Performance Dashboard**: Real-time display of prediction metrics and uncertainty.
- **Glassmorphic UI**: High-end aesthetic with vibrant gradients and micro-animations.

## 📊 Features

### 1. **Modular Architecture**
- Clean separation of concerns (config, models, utils, training)
- Easy to extend and modify
- Reusable components

### 2. **Comprehensive Configuration**
- Centralized configuration in `config.py`
- Easy to adjust hyperparameters
- Support for different model architectures

### 3. **Robust Data Processing**
- Automatic feature extraction (MFCC, Mel-spectrogram)
- Data augmentation support
- Efficient data loading with PyTorch DataLoader

### 4. **Advanced Models**
- **BiLSTM**: Bidirectional LSTM for sequence modeling
- **CNN**: Convolutional network for image processing
- **CombinedModel**: Hybrid CNN-BiLSTM architecture

### 5. **Training Features**
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Comprehensive logging
- Progress bars with tqdm

### 6. **Evaluation Metrics**
- Accuracy, F1-Score, Precision, Recall
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- Automatic threshold optimization

## 🔧 Configuration Options

Key parameters in `config.py`:

```python
# Audio Processing
SR = 16000                    # Sample rate
N_MFCC = 13                   # Number of MFCC coefficients
MAX_SEQ_LEN = 200             # Maximum sequence length

# Model Architecture
HIDDEN_DIM = 128              # LSTM hidden dimension
N_LAYERS = 2                  # Number of LSTM layers
BIDIRECTIONAL = True          # Use bidirectional LSTM
DROPOUT = 0.3                 # Dropout rate

# Training
BATCH_SIZE = 64               # Batch size
N_EPOCHS = 20                 # Number of epochs
LR = 1e-4                     # Learning rate
EARLY_STOPPING_PATIENCE = 5   # Early stopping patience
```

## 📈 Model Performance

The project tracks and logs:
- Training/validation loss and accuracy
- Learning rate changes
- Best model checkpoint
- Detailed evaluation metrics

Example output:
```
Epoch 1/20
------------------------------------------------------------
Training: 100%|██████████| 125/125 [00:15<00:00,  8.12it/s]
Validation: 100%|██████████| 32/32 [00:02<00:00, 14.56it/s]
Train Loss: 0.4523 | Train Acc: 78.45%
Val Loss: 0.3891 | Val Acc: 82.34%
Learning Rate: 0.000100
Saved best model with val_loss: 0.3891
```

## 🛠️ Advanced Usage

### Custom Model Architecture

You can modify the model architecture in `models.py`:

```python
# Example: Create a custom model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
```

### Data Augmentation

Add data augmentation in `utils.py`:

```python
def augment_audio(audio, sr):
    # Add noise, time shift, pitch shift, etc.
    return augmented_audio
```

### Custom Loss Functions

Modify the loss function in `train.py`:

```python
# Example: Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

## 📝 Logging

All training logs are saved to `logs/training.log` with:
- Timestamp
- Log level
- Detailed messages
- Training progress

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in `config.py`
   - Use CPU by setting `DEVICE = "cpu"` in `config.py`

2. **File Not Found Errors**
   - Check paths in `config.py`
   - Ensure data directory structure is correct

3. **Import Errors**
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Original dataset: TeamDeepwave
- PyTorch community
- Librosa for audio processing

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project has been refactored and improved for better usability, readability, and maintainability. The original code has been cleaned up, bugs fixed, and comprehensive documentation added.
