"""
Utilities for DeepWater

This module provides evaluation metrics, visualization tools, and other utilities.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_mape(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    return np.mean(np.abs((targets - predictions) / (np.abs(targets) + epsilon))) * 100


def compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(predictions, targets)[0, 1]


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metric names to values
    """
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "mape": compute_mape(predictions, targets),
        "r2": compute_r2(predictions, targets),
        "pearson_r": compute_pearson_correlation(predictions, targets),
        "min_error": np.min(np.abs(predictions - targets)),
        "max_error": np.max(np.abs(predictions - targets)),
        "std_error": np.std(predictions - targets),
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for name, value in metrics.items():
        if "r2" in name or "pearson" in name:
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value:.4f} ft")
    print(f"{'=' * 50}\n")


# ============================================================================
# Model Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    denormalize_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run on
        denormalize_fn: Function to denormalize predictions
        
    Returns:
        Tuple of (predictions, targets, metrics)
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Get predictions
        if "query" in batch:
            predictions = model(
                ref1=batch["ref1"],
                ref2=batch["ref2"],
                query=batch["query"],
                elevation1=batch["elevation1"],
                elevation2=batch["elevation2"],
            )
            targets = batch["elevation_query"]
        else:
            predictions = model(
                image1=batch["image1"],
                image2=batch["image2"],
                elevation1=batch["elevation1"],
            )
            targets = batch["elevation2"]
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Denormalize if needed
    if denormalize_fn is not None:
        predictions = denormalize_fn(predictions)
        targets = denormalize_fn(targets)
    
    metrics = compute_all_metrics(predictions, targets)
    
    return predictions, targets, metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    save_path: Optional[str] = None,
):
    """
    Plot predictions vs targets scatter plot.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    
    # Labels
    ax.set_xlabel("Target Water Level (ft)", fontsize=12)
    ax.set_ylabel("Predicted Water Level (ft)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics = compute_all_metrics(predictions, targets)
    text = f"MAE: {metrics['mae']:.3f} ft\nRMSE: {metrics['rmse']:.3f} ft\nR²: {metrics['r2']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    
    return fig, ax


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Error Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot error distribution histogram.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    errors = predictions - targets
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--", label="Zero error")
    ax.axvline(np.mean(errors), color="orange", linestyle="-", label=f"Mean: {np.mean(errors):.3f}")
    
    ax.set_xlabel("Prediction Error (ft)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
):
    """
    Plot training loss and metrics history.
    
    Args:
        history: Dictionary with train_loss, val_loss, etc.
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE plot
    ax = axes[1]
    if "train_mae" in history:
        ax.plot(epochs, history["train_mae"], label="Train MAE")
        ax.plot(epochs, history["val_mae"], label="Val MAE")
        ax.set_ylabel("MAE (ft)", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title("Mean Absolute Error", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig, axes


# ============================================================================
# Data utilities
# ============================================================================

def estimate_dataset_size(
    num_images: int,
    image_size: int = 224,
    channels: int = 3,
    dtype_bytes: int = 4,
) -> str:
    """
    Estimate memory requirements for a dataset.
    
    Args:
        num_images: Number of images
        image_size: Image dimensions (assuming square)
        channels: Number of channels
        dtype_bytes: Bytes per element (4 for float32)
        
    Returns:
        Human-readable size string
    """
    bytes_per_image = image_size * image_size * channels * dtype_bytes
    total_bytes = num_images * bytes_per_image
    
    if total_bytes < 1024**2:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024**3:
        return f"{total_bytes / 1024**2:.1f} MB"
    else:
        return f"{total_bytes / 1024**3:.1f} GB"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def format_parameters(count: int) -> str:
    """Format parameter count as human-readable string."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


# ============================================================================
# Image utilities
# ============================================================================

def is_valid_image(image_path: str) -> bool:
    """Check if an image file is valid and readable."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def get_image_stats(image_dir: str) -> Dict[str, any]:
    """
    Get statistics about images in a directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Dictionary with image statistics
    """
    from PIL import Image
    
    image_dir = Path(image_dir)
    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    
    image_files = [f for f in image_dir.iterdir() if f.suffix in extensions]
    
    sizes = []
    widths = []
    heights = []
    
    for img_path in image_files[:100]:  # Sample first 100
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes.append(img_path.stat().st_size)
        except Exception:
            continue
    
    return {
        "num_images": len(image_files),
        "avg_width": np.mean(widths) if widths else 0,
        "avg_height": np.mean(heights) if heights else 0,
        "avg_size_kb": np.mean(sizes) / 1024 if sizes else 0,
        "total_size_mb": sum(f.stat().st_size for f in image_files) / 1024**2,
    }
