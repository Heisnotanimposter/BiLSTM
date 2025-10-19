"""
Example usage scripts for BiLSTM Audio Classification
Demonstrates common use cases and workflows
"""

import numpy as np
import torch
from pathlib import Path

from config import CONFIG
from models import AudioClassifier, BiLSTM, CNN, CombinedModel
from utils import (
    load_file_paths,
    load_csv_labels,
    batch_extract_features,
    split_data,
    calculate_metrics,
    find_best_threshold,
    set_seed,
    count_parameters
)


def example_1_basic_training():
    """
    Example 1: Basic training workflow
    Shows how to extract features and train a model
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Training Workflow")
    print("=" * 60)
    
    # Setup
    set_seed(CONFIG.RANDOM_SEED)
    device = CONFIG.get_device()
    
    # 1. Load data
    print("\n1. Loading data...")
    file_paths, labels = load_csv_labels(str(CONFIG.TRAIN_CSV))
    print(f"   Loaded {len(file_paths)} files")
    
    # 2. Extract features
    print("\n2. Extracting MFCC features...")
    features = batch_extract_features(
        file_paths[:100],  # Use subset for demo
        feature_type="mfcc",
        sr=CONFIG.SR,
        n_mfcc=CONFIG.N_MFCC,
        max_seq_len=CONFIG.MAX_SEQ_LEN
    )
    print(f"   Extracted features shape: {features.shape}")
    
    # 3. Prepare labels
    print("\n3. Preparing labels...")
    labels_numeric = np.array([1 if label == 'fake' else 0 for label in labels[:100]])
    print(f"   Labels distribution: {np.bincount(labels_numeric)}")
    
    # 4. Split data
    print("\n4. Splitting data...")
    train_feat, val_feat, train_labels, val_labels = split_data(
        features, labels_numeric,
        test_size=CONFIG.TRAIN_VAL_SPLIT,
        random_state=CONFIG.RANDOM_SEED
    )
    print(f"   Train: {len(train_feat)}, Validation: {len(val_feat)}")
    
    # 5. Initialize model
    print("\n5. Initializing model...")
    model = AudioClassifier(
        input_dim=CONFIG.N_MFCC,
        hidden_dim=CONFIG.HIDDEN_DIM,
        n_layers=CONFIG.N_LAYERS,
        bidirectional=CONFIG.BIDIRECTIONAL,
        dropout=CONFIG.DROPOUT
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"   Model parameters: {num_params:,}")
    
    print("\n✓ Basic training workflow completed!")
    print("=" * 60 + "\n")


def example_2_model_architectures():
    """
    Example 2: Different model architectures
    Shows how to create and use different model types
    """
    print("\n" + "=" * 60)
    print("Example 2: Model Architectures")
    print("=" * 60)
    
    device = CONFIG.get_device()
    
    # 1. BiLSTM Model
    print("\n1. BiLSTM Model")
    bilstm_model = BiLSTM(
        input_dim=CONFIG.N_MFCC,
        hidden_dim=CONFIG.HIDDEN_DIM,
        output_dim=2,
        n_layers=CONFIG.N_LAYERS,
        bidirectional=CONFIG.BIDIRECTIONAL,
        dropout=CONFIG.DROPOUT
    ).to(device)
    print(f"   Parameters: {count_parameters(bilstm_model):,}")
    
    # 2. CNN Model
    print("\n2. CNN Model")
    cnn_model = CNN(
        output_dim=128,
        activation='gelu'
    ).to(device)
    print(f"   Parameters: {count_parameters(cnn_model):,}")
    
    # 3. Combined Model
    print("\n3. Combined CNN-BiLSTM Model")
    combined_model = CombinedModel(
        lstm_input_dim=CONFIG.N_MFCC,
        lstm_hidden_dim=CONFIG.HIDDEN_DIM,
        lstm_output_dim=128,
        lstm_n_layers=CONFIG.N_LAYERS,
        lstm_bidirectional=CONFIG.BIDIRECTIONAL,
        lstm_dropout=CONFIG.DROPOUT,
        cnn_output_dim=128,
        cnn_activation='gelu'
    ).to(device)
    print(f"   Parameters: {count_parameters(combined_model):,}")
    
    # 4. Test forward pass
    print("\n4. Testing forward pass...")
    batch_size = 4
    
    # Test BiLSTM
    mfcc_input = torch.randn(batch_size, CONFIG.N_MFCC, CONFIG.MAX_SEQ_LEN).to(device)
    bilstm_out = bilstm_model(mfcc_input.permute(0, 2, 1))
    print(f"   BiLSTM output shape: {bilstm_out.shape}")
    
    # Test CNN
    mel_input = torch.randn(batch_size, 1, 128, 128).to(device)
    cnn_out = cnn_model(mel_input)
    print(f"   CNN output shape: {cnn_out.shape}")
    
    # Test Combined
    combined_out = combined_model(mfcc_input, mel_input)
    print(f"   Combined output shape: {combined_out.shape}")
    
    print("\n✓ Model architectures tested successfully!")
    print("=" * 60 + "\n")


def example_3_evaluation_metrics():
    """
    Example 3: Evaluation metrics
    Shows how to calculate and interpret evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Example 3: Evaluation Metrics")
    print("=" * 60)
    
    # Simulate predictions
    print("\n1. Simulating predictions...")
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    
    print(f"   True labels distribution: {np.bincount(y_true)}")
    print(f"   Predicted labels distribution: {np.bincount(y_pred)}")
    
    # Calculate metrics
    print("\n2. Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print("\n3. Metrics Results:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Find best threshold
    print("\n4. Finding best threshold...")
    best_threshold, best_f1 = find_best_threshold(
        y_true, y_pred_proba,
        threshold_min=0.1,
        threshold_max=0.5,
        threshold_step=0.01
    )
    print(f"   Best threshold: {best_threshold:.4f}")
    print(f"   Best F1 score: {best_f1:.4f}")
    
    print("\n✓ Evaluation metrics calculated successfully!")
    print("=" * 60 + "\n")


def example_4_configuration():
    """
    Example 4: Configuration management
    Shows how to use and modify configuration
    """
    print("\n" + "=" * 60)
    print("Example 4: Configuration Management")
    print("=" * 60)
    
    # Print current configuration
    print("\n1. Current Configuration:")
    CONFIG.print_config()
    
    # Modify configuration
    print("\n2. Modifying configuration...")
    original_lr = CONFIG.LR
    CONFIG.LR = 1e-3
    print(f"   Changed learning rate from {original_lr} to {CONFIG.LR}")
    
    # Create directories
    print("\n3. Creating project directories...")
    CONFIG.create_directories()
    
    # Get device
    print("\n4. Device configuration:")
    device = CONFIG.get_device()
    print(f"   Using device: {device}")
    
    print("\n✓ Configuration management demonstrated!")
    print("=" * 60 + "\n")


def example_5_data_loading():
    """
    Example 5: Data loading utilities
    Shows different ways to load and process data
    """
    print("\n" + "=" * 60)
    print("Example 5: Data Loading Utilities")
    print("=" * 60)
    
    # 1. Load file paths
    print("\n1. Loading file paths...")
    if CONFIG.TRAIN_DIR.exists():
        file_paths = load_file_paths(str(CONFIG.TRAIN_DIR), "*.ogg")
        print(f"   Found {len(file_paths)} audio files")
    else:
        print(f"   Training directory not found at {CONFIG.TRAIN_DIR}")
    
    # 2. Load CSV labels
    print("\n2. Loading CSV labels...")
    if CONFIG.TRAIN_CSV.exists():
        file_paths, labels = load_csv_labels(str(CONFIG.TRAIN_CSV))
        print(f"   Loaded {len(file_paths)} files with labels")
        print(f"   Label distribution: {np.bincount([1 if l == 'fake' else 0 for l in labels])}")
    else:
        print(f"   Training CSV not found at {CONFIG.TRAIN_CSV}")
    
    # 3. Extract features (demonstration with dummy data)
    print("\n3. Feature extraction...")
    print("   (Skipping actual extraction for demo)")
    print("   Use: batch_extract_features(file_paths, feature_type='mfcc')")
    
    print("\n✓ Data loading utilities demonstrated!")
    print("=" * 60 + "\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("BiLSTM Audio Classification - Example Usage")
    print("=" * 60)
    
    examples = [
        ("Basic Training Workflow", example_1_basic_training),
        ("Model Architectures", example_2_model_architectures),
        ("Evaluation Metrics", example_3_evaluation_metrics),
        ("Configuration Management", example_4_configuration),
        ("Data Loading Utilities", example_5_data_loading),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

