# Quick Start Guide

Get up and running with the BiLSTM Audio Classification project in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation (5 minutes)

### 1. Clone or Navigate to Project
```bash
cd /Users/seungwonlee/BiLSTM
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration (2 minutes)

### 1. Update Paths in `config.py`

Open `config.py` and update these paths to match your setup:

```python
# Update these paths
DATASET_ROOT = DATA_DIR / "TeamDeepwave" / "dataset" / "open"
TRAIN_DIR = DATASET_ROOT / "train"
TRAIN_CSV = DATASET_ROOT / "train.csv"
```

### 2. Prepare Your Data

Make sure your data is organized like this:

```
data/
└── TeamDeepwave/
    └── dataset/
        └── open/
            ├── train/
            │   ├── audio1.ogg
            │   ├── audio2.ogg
            │   └── ...
            └── train.csv
```

Your `train.csv` should look like:
```csv
path,label
train/audio1.ogg,real
train/audio2.ogg,fake
...
```

## Training (10 minutes)

### Option 1: Basic Training
```bash
python train.py
```

This will:
- Extract MFCC features from audio files
- Split data into train/validation sets
- Train the model
- Save the best model to `models/best_model.pt`

### Option 2: Training with Preprocessed Features
```bash
# First run - extracts and saves features
python train.py

# Subsequent runs - uses saved features (faster)
python train.py --use-preprocessed
```

### Monitor Training

Watch the console output:
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

Check logs in `logs/training.log` for detailed information.

## Inference (1 minute)

### Predict Single File
```bash
python predict.py --model models/best_model.pt --input path/to/audio.ogg
```

Output:
```
============================================================
File: audio.ogg
Prediction: Fake
Confidence: 87.45%
============================================================
```

### Predict Multiple Files
```bash
python predict.py --model models/best_model.pt --input_dir path/to/audio/folder --output predictions.csv
```

### Predict from CSV
```bash
python predict.py --model models/best_model.pt --csv test.csv --output predictions.csv
```

## Examples (5 minutes)

Run the example script to see all features:

```bash
python example_usage.py
```

This demonstrates:
- Basic training workflow
- Different model architectures
- Evaluation metrics
- Configuration management
- Data loading utilities

## Common Commands

```bash
# Train model
python train.py

# Train with preprocessed features
python train.py --use-preprocessed

# Predict single file
python predict.py --input audio.ogg

# Predict directory
python predict.py --input_dir audio_folder

# Predict from CSV
python predict.py --csv test.csv

# Run examples
python example_usage.py
```

## Troubleshooting

### CUDA Out of Memory
Edit `config.py`:
```python
BATCH_SIZE = 32  # Reduce from 64
```

### File Not Found
Check paths in `config.py` and ensure data exists.

### Import Errors
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Experiment with Hyperparameters**
   - Edit `config.py` to change learning rate, batch size, etc.
   - Try different model architectures

2. **Improve Performance**
   - Add data augmentation
   - Try different features (Mel-spectrogram)
   - Use ensemble methods

3. **Deploy**
   - Create API endpoint
   - Build web interface
   - Containerize with Docker

## Getting Help

- Check `README.md` for detailed documentation
- Review `MAINTENANCE.md` for project improvements
- Run `python example_usage.py` for code examples
- Check `logs/training.log` for detailed logs

## Summary

You've successfully:
- ✅ Installed dependencies
- ✅ Configured paths
- ✅ Trained a model
- ✅ Made predictions
- ✅ Run examples

**Total time: ~20 minutes**

Enjoy using the BiLSTM Audio Classification project! 🎉

