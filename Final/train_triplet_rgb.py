#!/usr/bin/env python3
"""
Train Triplet model for water level estimation (RGB only).
Optimized for NVIDIA T4 GPU on Google Cloud Platform.

Usage:
    # In tmux session on GCP:
    python train_triplet_rgb.py --epochs 50 --batch_size 16 --lr 1e-4

    # With larger backbone:
    python train_triplet_rgb.py --epochs 50 --batch_size 8 --lr 5e-5 --backbone vit_small_patch16_224
"""

import os
import sys
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
from pathlib import Path
from datetime import datetime
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DEVICE AND PERFORMANCE SETUP
# =============================================================================

def setup_device():
    """Configure device and performance settings."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # T4 GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Check if T4 and adjust settings
        if "T4" in gpu_name:
            logger.info("Detected T4 GPU - using optimized settings")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (will be slow)")
    
    return device

DEVICE = setup_device()

# ImageNet normalization constants
MEAN_RGB = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_RGB = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =============================================================================
# MODEL
# =============================================================================

class TripletWaterLevelModel(nn.Module):
    """
    Triplet network with cross-attention for water level estimation.
    
    Architecture:
        - Shared ViT backbone extracts features from 3 images
        - Cross-attention: query attends to two reference images
        - Elevation embedding encodes known reference water levels
        - MLP head predicts query water level
    """

    def __init__(self, backbone='vit_tiny_patch16_224', hidden_dim=128, dropout=0.1):
        super().__init__()

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.elev_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ref1, ref2, query, elev1, elev2):
        feat1 = self.backbone(ref1)
        feat2 = self.backbone(ref2)
        feat_q = self.backbone(query)

        refs = torch.stack([feat1, feat2], dim=1)
        query_unsq = feat_q.unsqueeze(1)
        attn_out, _ = self.cross_attn(query_unsq, refs, refs)
        attn_out = attn_out.squeeze(1)

        elevs = torch.stack([elev1, elev2], dim=1)
        elev_emb = self.elev_embed(elevs)

        combined = torch.cat([feat_q, attn_out, elev_emb], dim=1)
        return self.head(combined).squeeze(-1)


# =============================================================================
# DATASET
# =============================================================================

def safe_load_image(path, size=(224, 224)):
    """Safely load image, return black image if corrupted."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return Image.new("RGB", size, (0, 0, 0))


class SameSiteTripletDataset(torch.utils.data.Dataset):
    """
    Triplet dataset with same-site constraint.
    
    Creates triplets (ref1, ref2, query) where all images are from the
    same camera site, preventing the model from learning site identification.
    """

    def __init__(self, csv_path, elev_mean=None, elev_std=None,
                 min_elev_diff=0.3, max_triplets=5000):
        
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")

        # Clean data
        self.df["gage_height_ft"] = pd.to_numeric(self.df["gage_height_ft"], errors="coerce")
        self.df = self.df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
        self.df = self.df[(self.df["gage_height_ft"] > 0) & 
                          (self.df["gage_height_ft"] < 50)].reset_index(drop=True)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if elev_mean is None:
            self.elev_mean = self.df["gage_height_ft"].mean()
            self.elev_std = self.df["gage_height_ft"].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std

        logger.info(f"Elevation stats: mean={self.elev_mean:.2f} ft, std={self.elev_std:.2f} ft")

        # Group by site
        self.site_groups = {}
        for site in self.df["camera_id"].unique():
            site_df = self.df[self.df["camera_id"] == site].reset_index(drop=True)
            if len(site_df) >= 3:
                self.site_groups[site] = site_df
        
        logger.info(f"Sites with 3+ images: {len(self.site_groups)}")

        # Build triplets
        self._build_triplets(min_elev_diff, max_triplets)

    def _build_triplets(self, min_elev_diff, max_triplets):
        """Build same-site triplets with elevation difference constraint."""
        self.triplets = []
        sites = list(self.site_groups.keys())

        np.random.seed(42)
        attempts = 0
        max_attempts = max_triplets * 20

        while len(self.triplets) < max_triplets and attempts < max_attempts:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            n = len(site_df)

            if n < 3:
                attempts += 1
                continue

            i, j, k = np.random.choice(n, 3, replace=False)
            elev_i = site_df.iloc[i]["gage_height_ft"]
            elev_j = site_df.iloc[j]["gage_height_ft"]

            if abs(elev_i - elev_j) >= min_elev_diff:
                self.triplets.append((site, i, j, k))

            attempts += 1

        logger.info(f"Created {len(self.triplets)} triplets")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]

        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]

        img1 = safe_load_image(row1["image_path"])
        img2 = safe_load_image(row2["image_path"])
        img3 = safe_load_image(row3["image_path"])

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)

        elev1 = (row1["gage_height_ft"] - self.elev_mean) / self.elev_std
        elev2 = (row2["gage_height_ft"] - self.elev_mean) / self.elev_std
        elev3 = (row3["gage_height_ft"] - self.elev_mean) / self.elev_std

        return {
            "ref1": img1,
            "ref2": img2,
            "query": img3,
            "elevation1": torch.tensor(elev1, dtype=torch.float32),
            "elevation2": torch.tensor(elev2, dtype=torch.float32),
            "elevation_query": torch.tensor(elev3, dtype=torch.float32),
        }


# =============================================================================
# METRICS AND VISUALIZATION
# =============================================================================

def compute_metrics(preds, targets):
    """Compute regression metrics."""
    preds, targets = np.array(preds), np.array(targets)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    pearson_r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(pearson_r),
    }


def plot_training_curves(train_losses, val_maes, save_path):
    """Plot training loss and validation MAE curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_maes, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MAE (ft)')
    ax2.set_title('Validation MAE')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(targets, preds, metrics, save_path):
    """Plot predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(targets, preds, alpha=0.5, s=20, c='#2196F3')
    
    min_val = min(targets.min(), preds.min()) - 0.5
    max_val = max(targets.max(), preds.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual Water Level (ft)', fontsize=12)
    ax.set_ylabel('Predicted Water Level (ft)', fontsize=12)
    ax.set_title(f'Triplet RGB Model - Test Results\n'
                 f'MAE: {metrics["mae"]:.3f} ft ({metrics["mae"]*12:.1f} in), '
                 f'R2: {metrics["r2"]:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def denorm_image(tensor):
    """Denormalize image tensor for visualization."""
    img = tensor.clone()
    img = img * STD_RGB + MEAN_RGB
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def save_sample_visualizations(samples, elev_mean, elev_std, output_dir, max_samples=4):
    """Save sample triplet visualizations."""
    for i, sample in enumerate(samples[:max_samples]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        ref1 = denorm_image(sample["ref1"])
        ref2 = denorm_image(sample["ref2"])
        query = denorm_image(sample["query"])
        
        pred_ft = sample["pred"] * elev_std + elev_mean
        target_ft = sample["target"] * elev_std + elev_mean
        
        axes[0].imshow(ref1)
        axes[0].set_title("Reference 1")
        axes[0].axis("off")
        
        axes[1].imshow(ref2)
        axes[1].set_title("Reference 2")
        axes[1].axis("off")
        
        axes[2].imshow(query)
        axes[2].set_title(f"Query\nTrue: {target_ft:.2f} ft\nPred: {pred_ft:.2f} ft")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i+1}.png", dpi=150)
        plt.close()


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """Main training function."""
    
    logger.info("=" * 70)
    logger.info("TRIPLET RGB MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Datasets
    logger.info("\nLoading datasets...")
    train_dataset = SameSiteTripletDataset(
        config["train_csv"],
        min_elev_diff=config["min_elev_diff"],
        max_triplets=config["max_triplets"],
    )

    val_dataset = SameSiteTripletDataset(
        config["val_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        min_elev_diff=config["min_elev_diff"],
        max_triplets=1500,
    )

    test_dataset = SameSiteTripletDataset(
        config["test_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        min_elev_diff=config["min_elev_diff"],
        max_triplets=1500,
    )

    # DataLoaders - optimized for T4
    num_workers = 4 if DEVICE.type == "cuda" else 0
    pin_memory = DEVICE.type == "cuda"
    
    # Triplet uses 3 images, so effective batch is 3x
    batch_size = max(1, config["batch_size"] // 2)
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Model
    logger.info("\nCreating model...")
    model = TripletWaterLevelModel(
        backbone=config["backbone"],
        hidden_dim=128,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # Mixed precision for T4
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"triplet_rgb_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    logger.info("\nStarting training...")
    best_val_mae = float("inf")
    train_losses = []
    val_maes = []
    
    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        for batch in pbar:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=True)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=True)
            query = batch["query"].to(DEVICE, non_blocking=True)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)
            targets = batch["elevation_query"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)
                loss = F.mse_loss(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

            pbar.set_postfix({"loss": f"{train_loss_sum / n_train:.4f}"})

        epoch_loss = train_loss_sum / n_train
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                ref1 = batch["ref1"].to(DEVICE, non_blocking=True)
                ref2 = batch["ref2"].to(DEVICE, non_blocking=True)
                query = batch["query"].to(DEVICE, non_blocking=True)
                elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
                elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    preds = model(ref1, ref2, query, elev1, elev2)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch["elevation_query"].numpy())

        # Denormalize
        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        
        scheduler.step()

        val_mae = val_metrics["mae"]
        val_maes.append(val_mae)

        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val MAE={val_mae:.3f} ft, R2={val_metrics['r2']:.3f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model_state_dict": model.state_dict(),
                "elev_mean": train_dataset.elev_mean,
                "elev_std": train_dataset.elev_std,
                "config": config,
                "epoch": epoch,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> Saved best model (MAE: {best_val_mae:.3f} ft)")

    # Save training curves
    plot_training_curves(train_losses, val_maes, output_dir / "training_curves.png")

    # Test evaluation
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.eval()

    test_preds, test_targets = [], []
    sample_buffer = []

    with torch.no_grad():
        for batch in test_loader:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=True)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=True)
            query = batch["query"].to(DEVICE, non_blocking=True)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)

            preds_np = preds.cpu().numpy()
            targets_np = batch["elevation_query"].numpy()

            test_preds.extend(preds_np)
            test_targets.extend(targets_np)

            # Store samples for visualization
            if len(sample_buffer) < 6:
                for i in range(min(ref1.size(0), 6 - len(sample_buffer))):
                    sample_buffer.append({
                        "ref1": batch["ref1"][i].cpu(),
                        "ref2": batch["ref2"][i].cpu(),
                        "query": batch["query"][i].cpu(),
                        "pred": preds_np[i],
                        "target": targets_np[i],
                    })

    # Compute final metrics
    test_preds_ft = np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    test_targets_ft = np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)

    # Save visualizations
    plot_predictions(test_targets_ft, test_preds_ft, test_metrics, 
                    output_dir / "test_predictions.png")
    save_sample_visualizations(sample_buffer, train_dataset.elev_mean, 
                               train_dataset.elev_std, output_dir)

    # Save results
    results = {
        "model": "triplet_rgb",
        "test_mae_ft": test_metrics["mae"],
        "test_mae_inches": test_metrics["mae"] * 12,
        "test_rmse_ft": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_pearson_r": test_metrics["pearson_r"],
        "best_epoch": checkpoint["epoch"],
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - TRIPLET RGB")
    logger.info("=" * 70)
    logger.info(f"Test MAE:  {test_metrics['mae']:.3f} ft ({test_metrics['mae']*12:.1f} inches)")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.3f} ft")
    logger.info(f"Test R2:   {test_metrics['r2']:.3f}")
    logger.info(f"Output:    {output_dir}")

    return test_metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Triplet RGB model for water level estimation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (will be halved for triplets)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224", 
                        choices=["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224"],
                        help="Backbone model")
    parser.add_argument("--train_csv", type=str, default="data/train_quality.csv")
    parser.add_argument("--val_csv", type=str, default="data/val_quality.csv")
    parser.add_argument("--test_csv", type=str, default="data/test_quality.csv")
    parser.add_argument("--max_triplets", type=int, default=5000, help="Max triplets for training")
    args = parser.parse_args()

    config = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "backbone": args.backbone,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "min_elev_diff": 0.3,
        "max_triplets": args.max_triplets,
    }

    train(config)
