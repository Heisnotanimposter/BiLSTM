"""
Configuration file for BiLSTM Audio Classification Project
Centralized configuration management for all scripts
"""

import os
from pathlib import Path


class Config:
    """Main configuration class for the project"""
    
    # ==================== Paths ====================
    # Base directories
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Dataset paths (update these to match your local setup)
    DATASET_ROOT = DATA_DIR / "TeamDeepwave" / "dataset" / "open"
    TRAIN_DIR = DATASET_ROOT / "train"
    TRAIN_CSV = DATASET_ROOT / "train.csv"
    TEST_CSV = DATASET_ROOT / "test.csv"
    
    # Preprocessed data paths
    PREPROCESSED_DIR = DATA_DIR / "preprocessed"
    TRAIN_MFCC_PATH = PREPROCESSED_DIR / "train_mfcc.npy"
    TRAIN_LABELS_PATH = PREPROCESSED_DIR / "train_labels.npy"
    VAL_MFCC_PATH = PREPROCESSED_DIR / "val_mfcc.npy"
    VAL_LABELS_PATH = PREPROCESSED_DIR / "val_labels.npy"
    
    # ==================== Audio Processing ====================
    # Sample rate for audio loading
    SR = 16000
    
    # MFCC parameters
    N_MFCC = 13  # Number of MFCC coefficients
    MAX_SEQ_LEN = 200  # Maximum sequence length for padding/truncation
    
    # Mel-spectrogram parameters
    N_MELS = 128  # Number of mel filter banks
    HOP_LENGTH = 512
    N_FFT = 2048
    
    # ==================== Model Architecture ====================
    # BiLSTM parameters
    HIDDEN_DIM = 128
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
    
    # CNN parameters (for mel-spectrogram)
    CNN_OUTPUT_DIM = 128
    
    # ==================== Training Parameters ====================
    BATCH_SIZE = 64
    N_EPOCHS = 20
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 0.001
    
    # Learning rate scheduling
    LR_SCHEDULER_STEP_SIZE = 5
    LR_SCHEDULER_GAMMA = 0.5
    
    # ==================== Data Parameters ====================
    TRAIN_VAL_SPLIT = 0.2  # 80% train, 20% validation
    RANDOM_SEED = 42
    
    # Number of classes
    N_CLASSES = 2
    
    # ==================== Text Processing (for text-based BiLSTM) ====================
    # Embedding parameters
    EMBEDDING_DIM = 300
    VOCAB_MIN_FREQ = 3
    MAX_VOCAB_SIZE = 100000
    
    # ==================== Device Configuration ====================
    DEVICE = "cuda"  # Will be set to "cpu" if CUDA is not available
    NUM_WORKERS = 4  # Number of workers for DataLoader
    
    # ==================== Logging and Output ====================
    LOG_LEVEL = "INFO"
    SAVE_PREDICTIONS = True
    SAVE_MODEL_CHECKPOINTS = True
    
    # ==================== Evaluation ====================
    EVAL_THRESHOLD_MIN = 0.1
    EVAL_THRESHOLD_MAX = 0.5
    EVAL_THRESHOLD_STEP = 0.01
    
    # ==================== Methods ====================
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR,
            cls.PREPROCESSED_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directories: {[str(d) for d in directories]}")
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device (CUDA or CPU)"""
        import torch
        if torch.cuda.is_available() and cls.DEVICE == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("BiLSTM Audio Classification - Configuration")
        print("=" * 60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Dataset Root: {cls.DATASET_ROOT}")
        print(f"Sample Rate: {cls.SR}")
        print(f"MFCC Coefficients: {cls.N_MFCC}")
        print(f"Max Sequence Length: {cls.MAX_SEQ_LEN}")
        print(f"Hidden Dimension: {cls.HIDDEN_DIM}")
        print(f"Number of Layers: {cls.N_LAYERS}")
        print(f"Bidirectional: {cls.BIDIRECTIONAL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Number of Epochs: {cls.N_EPOCHS}")
        print(f"Learning Rate: {cls.LR}")
        print(f"Random Seed: {cls.RANDOM_SEED}")
        print(f"Device: {cls.get_device()}")
        print("=" * 60)


# Create a global config instance
CONFIG = Config()

