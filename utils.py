"""
Utility functions for BiLSTM Audio Classification Project
Includes data loading, preprocessing, logging, and evaluation utilities
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==================== Logging Setup ====================
def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
    # Create logger
    logger = logging.getLogger("BiLSTM")
    logger.setLevel(getattr(logging, log_level))
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==================== Data Loading ====================
def load_file_paths(root_folder: str, extension: str = "*.ogg") -> List[str]:
    """
    Load file paths from a directory
    
    Args:
        root_folder: Root directory containing files
        extension: File extension pattern (e.g., "*.ogg", "*.wav")
    
    Returns:
        List of file paths
    """
    search_pattern = os.path.join(root_folder, extension)
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"Warning: No files found matching pattern: {search_pattern}")
    
    return sorted(file_paths)


def load_csv_labels(csv_path: str, path_column: str = "path", 
                    label_column: str = "label") -> Tuple[List[str], List[int]]:
    """
    Load file paths and labels from a CSV file
    
    Args:
        csv_path: Path to CSV file
        path_column: Name of the column containing file paths
        label_column: Name of the column containing labels
    
    Returns:
        Tuple of (file_paths, labels)
    """
    df = pd.read_csv(csv_path)
    
    if path_column not in df.columns:
        raise ValueError(f"Column '{path_column}' not found in CSV")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in CSV")
    
    file_paths = df[path_column].tolist()
    labels = df[label_column].tolist()
    
    return file_paths, labels


# ==================== Audio Feature Extraction ====================
def extract_mfcc_features(file_path: str, sr: int, n_mfcc: int, 
                         max_seq_len: int) -> Optional[np.ndarray]:
    """
    Extract MFCC features from an audio file
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        max_seq_len: Maximum sequence length for padding/truncation
    
    Returns:
        MFCC features array of shape (n_mfcc, max_seq_len) or None if error
    """
    try:
        # Load audio file
        y, _ = librosa.load(file_path, sr=sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        pad_width = max_seq_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_seq_len]
        
        return mfcc
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_mel_spectrogram(file_path: str, sr: int, n_mels: int,
                            hop_length: int, n_fft: int) -> Optional[np.ndarray]:
    """
    Extract Mel-spectrogram from an audio file
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel filter banks
        hop_length: Hop length for STFT
        n_fft: FFT window size
    
    Returns:
        Mel-spectrogram array or None if error
    """
    try:
        # Load audio file
        y, _ = librosa.load(file_path, sr=sr)
        
        # Extract Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, 
            hop_length=hop_length, n_fft=n_fft
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def batch_extract_features(file_paths: List[str], feature_type: str = "mfcc",
                          sr: int = 16000, n_mfcc: int = 13, 
                          max_seq_len: int = 200, n_mels: int = 128,
                          hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    """
    Extract features from multiple audio files
    
    Args:
        file_paths: List of audio file paths
        feature_type: Type of features to extract ("mfcc" or "mel")
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        max_seq_len: Maximum sequence length
        n_mels: Number of mel filter banks
        hop_length: Hop length for STFT
        n_fft: FFT window size
    
    Returns:
        Array of features
    """
    features = []
    
    for file_path in tqdm(file_paths, desc=f"Extracting {feature_type} features"):
        if feature_type == "mfcc":
            feature = extract_mfcc_features(file_path, sr, n_mfcc, max_seq_len)
        elif feature_type == "mel":
            feature = extract_mel_spectrogram(file_path, sr, n_mels, hop_length, n_fft)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        if feature is not None:
            features.append(feature)
    
    return np.array(features)


# ==================== Data Splitting ====================
def split_data(features: np.ndarray, labels: np.ndarray,
               test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into training and validation sets
    
    Args:
        features: Feature array
        labels: Label array
        test_size: Proportion of data for validation
        random_state: Random seed
    
    Returns:
        Tuple of (train_features, val_features, train_labels, val_labels)
    """
    return train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state
    )


# ==================== Model Evaluation ====================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_pred_proba: Optional[np.ndarray] = None) -> dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def find_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       threshold_min: float = 0.1, threshold_max: float = 0.5,
                       threshold_step: float = 0.01) -> Tuple[float, float]:
    """
    Find the best threshold for binary classification based on F1 score
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold_min: Minimum threshold to test
        threshold_max: Maximum threshold to test
        threshold_step: Step size for threshold search
    
    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    best_threshold = threshold_min
    best_f1 = 0.0
    
    for threshold in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str] = None) -> None:
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
    """
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 60 + "\n")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str] = None) -> None:
    """
    Print confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    print(cm)
    print("=" * 60 + "\n")


# ==================== Model Saving/Loading ====================
def save_model(model: torch.nn.Module, save_path: Path, 
              additional_info: dict = None) -> None:
    """
    Save model and additional information
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        additional_info: Additional information to save (optional)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), save_path)
    
    # Save additional info if provided
    if additional_info:
        info_path = save_path.parent / f"{save_path.stem}_info.pt"
        torch.save(additional_info, info_path)


def load_model(model: torch.nn.Module, load_path: Path) -> torch.nn.Module:
    """
    Load model from file
    
    Args:
        model: PyTorch model instance
        load_path: Path to load model from
    
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    return model


# ==================== Utility Functions ====================
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

