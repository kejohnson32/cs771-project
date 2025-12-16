#!/usr/bin/env python3
"""
Training Script for Water Level Estimation

This script provides a complete training pipeline that can be run 
on Google Colab or any GPU machine.

Usage:
    python train.py --config configs/default.yaml
    
    # Or with command-line arguments:
    python train.py --data data/train.csv --images-dir data/images \
                    --model-type siamese --epochs 100
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepwater.data import (
    WaterLevelDataset,
    MultiSiteDataset,
    split_dataset,
    create_dataloaders,
)
from deepwater.models import (
    create_model,
    create_model_from_config,
    MODEL_CONFIGS,
)
from deepwater.training import (
    WaterLevelTrainer,
    create_optimizer,
    create_scheduler,
)
from deepwater.utils import (
    evaluate_model,
    print_metrics,
    plot_predictions,
    plot_training_history,
    count_parameters,
    format_parameters,
)
from deepwater.configs import Config, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_training(args):
    """Setup training configuration and data."""
    
    # Load config if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command-line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.model_type:
        config.model.model_type = args.model_type
    if args.backbone:
        config.model.backbone = args.backbone
    
    return config


def prepare_data(args, config):
    """Prepare datasets and dataloaders."""
    
    # Check if we need to split
    data_path = Path(args.data)
    
    if data_path.is_file():
        # Single CSV - need to split
        logger.info(f"Splitting dataset: {data_path}")
        
        output_dir = data_path.parent / "splits"
        train_csv, val_csv, test_csv = split_dataset(
            str(data_path),
            str(output_dir),
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            test_ratio=config.training.test_ratio,
            stratify_by="00065",
        )
    else:
        # Assume pre-split directory
        train_csv = str(data_path / "train.csv")
        val_csv = str(data_path / "val.csv")
        test_csv = str(data_path / "test.csv")
    
    # Create dataloaders
    mode = config.model.model_type
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        images_dir=args.images_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        mode=mode,
        image_size=config.model.image_size,
    )
    
    logger.info(f"Dataset sizes: Train={len(train_loader.dataset)}, "
                f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def create_model_and_optimizer(config, train_loader):
    """Create model, optimizer, and scheduler."""
    
    # Create model
    if config.model.model_type in MODEL_CONFIGS:
        model = create_model_from_config(config.model.model_type)
    else:
        model = create_model(
            model_type=config.model.model_type,
            backbone=config.model.backbone,
            pretrained=config.model.pretrained,
            feature_dim=config.model.embed_dim,
            hidden_dim=config.model.projection_dim,
            dropout=config.model.dropout,
        )
    
    # Log model info
    params = count_parameters(model)
    logger.info(f"Model: {config.model.model_type} with {config.model.backbone}")
    logger.info(f"Parameters: {format_parameters(params['total'])} "
                f"({format_parameters(params['trainable'])} trainable)")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.lr_scheduler,
        num_epochs=config.training.num_epochs,
        warmup_epochs=config.training.warmup_epochs,
        min_lr=config.training.min_lr,
    )
    
    return model, optimizer, scheduler


def train(args):
    """Main training function."""
    
    # Setup
    config = setup_training(args)
    
    # Device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader = prepare_data(args, config)
    
    # Model
    model, optimizer, scheduler = create_model_and_optimizer(config, train_loader)
    
    # Experiment name
    experiment_name = args.experiment_name or (
        f"{config.model.model_type}_{config.model.backbone.split('_')[0]}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create trainer
    trainer = WaterLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        use_amp=config.training.use_amp,
        gradient_clip=config.training.gradient_clip,
        log_interval=config.training.log_every_n_steps,
        save_interval=config.training.save_every_n_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        use_wandb=config.training.use_wandb,
        wandb_project=config.training.wandb_project,
    )
    
    # Save config
    config_path = Path(args.output_dir) / experiment_name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_path))
    
    # Train
    logger.info(f"Starting training for {config.training.num_epochs} epochs")
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from=args.resume,
    )
    
    # Plot training history
    plot_training_history(
        history,
        title=f"Training History - {experiment_name}",
        save_path=str(Path(args.output_dir) / experiment_name / "training_history.png"),
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    
    model_path = Path(args.output_dir) / experiment_name / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    predictions, targets, metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        denormalize_fn=train_loader.dataset.denormalize_elevation,
    )
    
    print_metrics(metrics, title="Test Set Results")
    
    # Save test results
    results_path = Path(args.output_dir) / experiment_name / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot predictions
    plot_predictions(
        predictions, targets,
        title=f"Test Predictions - {experiment_name}",
        save_path=str(Path(args.output_dir) / experiment_name / "test_predictions.png"),
    )
    
    logger.info(f"Training complete! Results saved to: {args.output_dir}/{experiment_name}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train water level estimation model")
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data CSV or splits directory")
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Directory containing images")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--experiment-name", type=str,
                       help="Experiment name (default: auto-generated)")
    
    # Config
    parser.add_argument("--config", type=str,
                       help="Path to config YAML file")
    
    # Model arguments (override config)
    parser.add_argument("--model-type", type=str, choices=["siamese", "triplet"],
                       help="Model type")
    parser.add_argument("--backbone", type=str,
                       help="Backbone model name")
    
    # Training arguments (override config)
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    # Resume training
    parser.add_argument("--resume", type=str,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
