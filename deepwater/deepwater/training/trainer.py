"""
Training Module for Water Level Estimation

This module provides the training loop, metrics, and utilities
for training water level estimation models.
"""

import os
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Callable, Any, Union
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR,
)
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    train_rmse: float
    val_rmse: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class WarmupScheduler:
    """Learning rate warmup wrapper."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Any,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr * warmup_factor
        else:
            self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class WaterLevelTrainer:
    """
    Trainer for water level estimation models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "auto",
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 5,
        early_stopping_patience: int = 15,
        use_wandb: bool = False,
        wandb_project: str = "deepwater",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            device: Device to train on ("auto", "cuda", "mps", "cpu")
            output_dir: Directory for outputs
            experiment_name: Name for this experiment
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping value
            log_interval: Steps between logging
            save_interval: Epochs between checkpoints
            early_stopping_patience: Patience for early stopping
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Training settings
        self.use_amp = use_amp and self.device.type == "cuda"
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Output directory
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=self.experiment_name)
                wandb.watch(self.model)
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B logging.")
                self.use_wandb = False
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        self.current_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, mae, rmse)
        """
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_samples = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions, targets, loss = self._forward_batch(batch)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions, targets, loss = self._forward_batch(batch)
                
                # Backward pass
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                
                self.optimizer.step()
            
            # Compute metrics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_mae += F.l1_loss(predictions, targets, reduction="sum").item()
            total_mse += F.mse_loss(predictions, targets, reduction="sum").item()
            num_samples += batch_size
            
            # Update progress bar
            if (step + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": total_loss / num_samples,
                    "mae": total_mae / num_samples,
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        avg_rmse = math.sqrt(total_mse / num_samples)
        
        return avg_loss, avg_mae, avg_rmse
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """
        Run validation.
        
        Returns:
            Tuple of (loss, mae, rmse)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions, targets, loss = self._forward_batch(batch)
            else:
                predictions, targets, loss = self._forward_batch(batch)
            
            # Compute metrics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_mae += F.l1_loss(predictions, targets, reduction="sum").item()
            total_mse += F.mse_loss(predictions, targets, reduction="sum").item()
            num_samples += batch_size
        
        # Compute epoch metrics
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        avg_rmse = math.sqrt(total_mse / num_samples)
        
        return avg_loss, avg_mae, avg_rmse
    
    def _forward_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch.
        
        Handles both siamese and triplet models.
        
        Returns:
            Tuple of (predictions, targets, loss)
        """
        # Check which type of batch this is
        if "query" in batch:
            # Triplet model
            predictions = self.model(
                ref1=batch["ref1"],
                ref2=batch["ref2"],
                query=batch["query"],
                elevation1=batch["elevation1"],
                elevation2=batch["elevation2"],
            )
            targets = batch["elevation_query"]
        else:
            # Siamese model
            predictions = self.model(
                image1=batch["image1"],
                image2=batch["image2"],
                elevation1=batch["elevation1"],
            )
            targets = batch["elevation2"]
        
        # Compute loss
        loss = F.mse_loss(predictions, targets)
        
        return predictions, targets, loss
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Dictionary of metric histories
        """
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch()
            
            # Validate
            val_loss, val_mae, val_rmse = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
            self.metrics_history.append(metrics)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # W&B logging
            if self.use_wandb:
                import wandb
                wandb.log(metrics.to_dict())
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", is_best=True)
                logger.info(f"New best model! Val Loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save metrics history
        self._save_metrics_history()
        
        logger.info("Training complete!")
        
        return {
            "train_loss": [m.train_loss for m in self.metrics_history],
            "val_loss": [m.val_loss for m in self.metrics_history],
            "train_mae": [m.train_mae for m in self.metrics_history],
            "val_mae": [m.val_mae for m in self.metrics_history],
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics_history": [asdict(m) for m in self.metrics_history],
        }
        
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            try:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            except:
                pass
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if "metrics_history" in checkpoint:
            self.metrics_history = [
                TrainingMetrics(**m) for m in checkpoint["metrics_history"]
            ]
        
        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")
    
    def _save_metrics_history(self):
        """Save metrics history to JSON."""
        history = [asdict(m) for m in self.metrics_history]
        path = self.output_dir / "metrics_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: "adamw", "adam", or "sgd"
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type == "adamw":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "adam":
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    steps_per_epoch: Optional[int] = None,
    **kwargs,
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: "cosine", "step", "plateau", or "onecycle"
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "step":
        base_scheduler = StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            min_lr=min_lr,
        )
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    # Wrap with warmup
    if warmup_epochs > 0:
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    
    return base_scheduler
