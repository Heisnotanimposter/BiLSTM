"""
Main training script for BiLSTM Audio Classification
Refactored and improved version of the original bilstmtest.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from config import CONFIG
from models import AudioClassifier
from utils import (
    setup_logging,
    load_file_paths,
    load_csv_labels,
    batch_extract_features,
    split_data,
    save_model,
    load_model,
    calculate_metrics,
    find_best_threshold,
    print_classification_report,
    print_confusion_matrix,
    set_seed,
    count_parameters,
    format_time
)


class AudioDataset(Dataset):
    """Custom Dataset for audio features"""
    
    def __init__(self, features, labels=None):
        """
        Initialize dataset
        
        Args:
            features: Audio features array
            labels: Labels array (optional for inference)
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]]).squeeze()
            return feature, label
        return feature


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda or cpu)
        logger: Logger instance
    
    Returns:
        Average training loss and accuracy
    """
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, logger):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        logger: Logger instance
    
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    avg_loss = epoch_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, logger, n_epochs):
    """
    Main training loop
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda or cpu)
        logger: Logger instance
        n_epochs: Number of training epochs
    
    Returns:
        Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{n_epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device, logger
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            model_path = CONFIG.MODELS_DIR / "best_model.pt"
            save_model(model, model_path, {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    return history


def main(args):
    """Main function"""
    # Setup
    set_seed(CONFIG.RANDOM_SEED)
    CONFIG.create_directories()
    logger = setup_logging(CONFIG.LOGS_DIR, CONFIG.LOG_LEVEL)
    device = CONFIG.get_device()
    
    logger.info("Starting BiLSTM Audio Classification Training")
    CONFIG.print_config()
    
    # Check if data exists
    if not CONFIG.TRAIN_CSV.exists():
        logger.error(f"Training CSV not found at {CONFIG.TRAIN_CSV}")
        logger.info("Please update CONFIG.TRAIN_CSV in config.py with the correct path")
        return
    
    # Load data
    logger.info("Loading data...")
    if args.use_preprocessed:
        # Load preprocessed features
        logger.info("Loading preprocessed features...")
        train_features = np.load(CONFIG.TRAIN_MFCC_PATH)
        train_labels = np.load(CONFIG.TRAIN_LABELS_PATH)
        val_features = np.load(CONFIG.VAL_MFCC_PATH)
        val_labels = np.load(CONFIG.VAL_LABELS_PATH)
    else:
        # Extract features from audio files
        logger.info("Extracting features from audio files...")
        file_paths, labels = load_csv_labels(str(CONFIG.TRAIN_CSV))
        
        # Extract MFCC features
        features = batch_extract_features(
            file_paths,
            feature_type="mfcc",
            sr=CONFIG.SR,
            n_mfcc=CONFIG.N_MFCC,
            max_seq_len=CONFIG.MAX_SEQ_LEN
        )
        
        # Convert labels to numeric
        labels = np.array([1 if label == 'fake' else 0 for label in labels])
        
        # Split data
        train_features, val_features, train_labels, val_labels = split_data(
            features, labels,
            test_size=CONFIG.TRAIN_VAL_SPLIT,
            random_state=CONFIG.RANDOM_SEED
        )
        
        # Save preprocessed data
        logger.info("Saving preprocessed features...")
        np.save(CONFIG.TRAIN_MFCC_PATH, train_features)
        np.save(CONFIG.TRAIN_LABELS_PATH, train_labels)
        np.save(CONFIG.VAL_MFCC_PATH, val_features)
        np.save(CONFIG.VAL_LABELS_PATH, val_labels)
    
    logger.info(f"Training samples: {len(train_features)}")
    logger.info(f"Validation samples: {len(val_features)}")
    
    # Create datasets
    train_dataset = AudioDataset(train_features, train_labels)
    val_dataset = AudioDataset(val_features, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        num_workers=CONFIG.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = AudioClassifier(
        input_dim=CONFIG.N_MFCC,
        hidden_dim=CONFIG.HIDDEN_DIM,
        n_layers=CONFIG.N_LAYERS,
        bidirectional=CONFIG.BIDIRECTIONAL,
        dropout=CONFIG.DROPOUT
    ).to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG.LR,
        weight_decay=CONFIG.WEIGHT_DECAY
    )
    scheduler = StepLR(
        optimizer,
        step_size=CONFIG.LR_SCHEDULER_STEP_SIZE,
        gamma=CONFIG.LR_SCHEDULER_GAMMA
    )
    
    # Train model
    logger.info("Starting training...")
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, logger, CONFIG.N_EPOCHS
    )
    
    # Load best model and evaluate
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating best model...")
    logger.info("=" * 60)
    
    model_path = CONFIG.MODELS_DIR / "best_model.pt"
    model.load_state_dict(torch.load(model_path))
    
    # Final evaluation
    val_loss, val_acc, val_preds, val_labels, val_probs = validate(
        model, val_loader, criterion, device, logger
    )
    
    # Calculate metrics
    metrics = calculate_metrics(val_labels, val_preds, val_probs)
    logger.info(f"Final Validation Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Find best threshold
    best_threshold, best_f1 = find_best_threshold(
        val_labels, val_probs,
        threshold_min=CONFIG.EVAL_THRESHOLD_MIN,
        threshold_max=CONFIG.EVAL_THRESHOLD_MAX,
        threshold_step=CONFIG.EVAL_THRESHOLD_STEP
    )
    logger.info(f"Best threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
    
    # Print classification report
    print_classification_report(val_labels, val_preds, ['Real', 'Fake'])
    print_confusion_matrix(val_labels, val_preds, ['Real', 'Fake'])
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM Audio Classifier")
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Use preprocessed features instead of extracting from audio files"
    )
    args = parser.parse_args()
    
    main(args)

