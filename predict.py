"""
Inference script for BiLSTM Audio Classification
Loads a trained model and makes predictions on new audio files
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from config import CONFIG
from models import AudioClassifier
from utils import (
    setup_logging,
    load_model,
    extract_mfcc_features,
    set_seed
)


def predict_single_audio(model, audio_path, device, logger):
    """
    Predict class for a single audio file
    
    Args:
        model: Trained PyTorch model
        audio_path: Path to audio file
        device: Device (cuda or cpu)
        logger: Logger instance
    
    Returns:
        Predicted class and probability
    """
    # Extract MFCC features
    features = extract_mfcc_features(
        str(audio_path),
        sr=CONFIG.SR,
        n_mfcc=CONFIG.N_MFCC,
        max_seq_len=CONFIG.MAX_SEQ_LEN
    )
    
    if features is None:
        logger.error(f"Failed to extract features from {audio_path}")
        return None, None
    
    # Convert to tensor and add batch dimension
    features = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Map class index to label
    class_labels = ['Real', 'Fake']
    predicted_label = class_labels[predicted_class]
    
    return predicted_label, confidence


def predict_batch(model, audio_paths, device, logger):
    """
    Predict classes for multiple audio files
    
    Args:
        model: Trained PyTorch model
        audio_paths: List of audio file paths
        device: Device (cuda or cpu)
        logger: Logger instance
    
    Returns:
        List of (predicted_label, confidence) tuples
    """
    results = []
    
    for audio_path in tqdm(audio_paths, desc="Predicting"):
        predicted_label, confidence = predict_single_audio(
            model, audio_path, device, logger
        )
        
        if predicted_label is not None:
            results.append({
                'file': audio_path.name,
                'prediction': predicted_label,
                'confidence': confidence
            })
    
    return results


def main(args):
    """Main function"""
    # Setup
    set_seed(CONFIG.RANDOM_SEED)
    CONFIG.create_directories()
    logger = setup_logging(CONFIG.LOGS_DIR, CONFIG.LOG_LEVEL)
    device = CONFIG.get_device()
    
    logger.info("Starting BiLSTM Audio Classification Inference")
    logger.info(f"Using device: {device}")
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = AudioClassifier(
        input_dim=CONFIG.N_MFCC,
        hidden_dim=CONFIG.HIDDEN_DIM,
        n_layers=CONFIG.N_LAYERS,
        bidirectional=CONFIG.BIDIRECTIONAL,
        dropout=CONFIG.DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Process input
    if args.input:
        # Single file prediction
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
        
        logger.info(f"Predicting on: {input_path}")
        predicted_label, confidence = predict_single_audio(
            model, input_path, device, logger
        )
        
        if predicted_label:
            print("\n" + "=" * 60)
            print(f"File: {input_path.name}")
            print(f"Prediction: {predicted_label}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("=" * 60 + "\n")
    
    elif args.input_dir:
        # Batch prediction
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return
        
        # Find all audio files
        audio_extensions = ['*.ogg', '*.wav', '*.mp3', '*.flac']
        audio_paths = []
        for ext in audio_extensions:
            audio_paths.extend(input_dir.glob(ext))
        
        if not audio_paths:
            logger.error(f"No audio files found in {input_dir}")
            return
        
        logger.info(f"Found {len(audio_paths)} audio files")
        
        # Make predictions
        results = predict_batch(model, audio_paths, device, logger)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        # Print results
        print("\n" + "=" * 60)
        print("Prediction Results")
        print("=" * 60)
        for result in results:
            print(f"{result['file']}: {result['prediction']} ({result['confidence']*100:.2f}%)")
        print("=" * 60 + "\n")
    
    elif args.csv:
        # Predict from CSV file
        import pandas as pd
        
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        if 'path' not in df.columns:
            logger.error("CSV must have a 'path' column")
            return
        
        # Get base directory for paths
        base_dir = csv_path.parent if not args.base_dir else Path(args.base_dir)
        
        # Load audio paths
        audio_paths = [base_dir / path for path in df['path']]
        
        logger.info(f"Predicting on {len(audio_paths)} files from CSV")
        
        # Make predictions
        results = predict_batch(model, audio_paths, device, logger)
        
        # Add predictions to dataframe
        df['prediction'] = [r['prediction'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            # Save to same directory with _predictions suffix
            output_path = csv_path.parent / f"{csv_path.stem}_predictions.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        print("\n" + "=" * 60)
        print(f"Predictions completed for {len(results)} files")
        print("=" * 60 + "\n")
    
    else:
        logger.error("Please specify --input, --input_dir, or --csv")
        return
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BiLSTM Audio Classification Inference"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to single audio file"
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing audio files"
    )
    input_group.add_argument(
        "--csv",
        type=str,
        help="CSV file with audio file paths"
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Base directory for paths in CSV (if not absolute)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for predictions (CSV)"
    )
    
    args = parser.parse_args()
    
    main(args)

