#!/usr/bin/env python3
"""
Train Triplet model for water level estimation (RGB only).

Usage:
    python train_triplet_5k.py --epochs 30 --batch_size 8 --lr 1e-4
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL PERFORMANCE & DEVICE SETUP
# =============================================================================

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f" Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(" Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print(" Using CPU")

# For de-normalization of 3-channel images (ImageNet stats)
MEAN_3 = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_3 = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =============================================================================
# MODEL
# =============================================================================

class TripletWaterLevelModel(nn.Module):
    """Triplet network with cross-attention for water level estimation."""

    def __init__(
        self,
        backbone: str = "vit_tiny_patch16_224",
        feature_dim: int = 192,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]

        # Cross-attention between references and query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Elevation embeddings
        self.elev_embed = nn.Sequential(
            nn.Linear(2, 64),  # 2 reference elevations
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Prediction head: query_feat + attn_feat + elev_embed
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ref1, ref2, query, elev1, elev2):
        """
        Args:
            ref1, ref2: Reference images [B, 3, 224, 224]
            query: Query image [B, 3, 224, 224]
            elev1, elev2: Reference elevations (normalized) [B]
        Returns:
            Predicted elevation for query [B]
        """
        # Extract features
        feat1 = self.backbone(ref1)   # [B, D]
        feat2 = self.backbone(ref2)   # [B, D]
        feat_q = self.backbone(query) # [B, D]

        # Stack references: [B, 2, D]
        refs = torch.stack([feat1, feat2], dim=1)

        # Cross-attention: query attends to refs
        query_unsq = feat_q.unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.cross_attn(query_unsq, refs, refs)
        attn_out = attn_out.squeeze(1)    # [B, D]

        # Elevation embedding
        elevs = torch.stack([elev1, elev2], dim=1)  # [B, 2]
        elev_emb = self.elev_embed(elevs)           # [B, 64]

        # Combine and predict
        combined = torch.cat([feat_q, attn_out, elev_emb], dim=1)
        pred = self.head(combined).squeeze(-1)
        return pred


def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        print(f"[WARN] Bad image, using blank instead: {path} ({e})")
        # create a dummy black image; transforms will resize it anyway
        return Image.new("RGB", (512, 512), (0, 0, 0))
    except Exception as e:
        print(f"[WARN] Error opening image {path}: {e}")
        return Image.new("RGB", (512, 512), (0, 0, 0))


# =============================================================================
# DATASET
# =============================================================================

class SameSiteTripletDataset(torch.utils.data.Dataset):
    """Triplet dataset with same-site constraint."""

    def __init__(
        self,
        csv_path,
        elev_mean=None,
        elev_std=None,
        min_elev_diff: float = 0.3,
        max_triplets: int = 5000,
    ):
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Clean data
        self.df["gage_height_ft"] = pd.to_numeric(
            self.df["gage_height_ft"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
        self.df = self.df[
            (self.df["gage_height_ft"] > 0)
            & (self.df["gage_height_ft"] < 50)
        ].reset_index(drop=True)

        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if elev_mean is None:
            self.elev_mean = self.df["gage_height_ft"].mean()
            self.elev_std = self.df["gage_height_ft"].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std

        print(
            f"Elevation normalization: mean={self.elev_mean:.2f}, "
            f"std={self.elev_std:.2f}"
        )

        # Group by site (camera_id)
        self.site_groups = {}
        for site in self.df["camera_id"].unique():
            site_df = self.df[self.df["camera_id"] == site].reset_index(drop=True)
            if len(site_df) >= 3:  # Need at least 3 for triplet
                self.site_groups[site] = site_df
        print(f"Sites with 3+ images: {len(self.site_groups)}")

        # Build triplets (same site only)
        self.triplets = []
        sites = list(self.site_groups.keys())

        np.random.seed(42)
        attempts = 0
        while len(self.triplets) < max_triplets and attempts < max_triplets * 20:
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

        print(f"Created {len(self.triplets)} same-site triplets")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]

        row1 = site_df.iloc[i]
        row2 = site_df.iloc[j]
        row3 = site_df.iloc[k]

        img1 = safe_open_image(row1["image_path"])
        img2 = safe_open_image(row2["image_path"])
        img3 = safe_open_image(row3["image_path"])

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
# METRICS
# =============================================================================

def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": corr,
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def denorm_3ch(img_3ch: torch.Tensor):
    """
    Inverse normalization for a 3-channel tensor.
    Returns: rgb_np (H, W, 3)
    """
    x = img_3ch.clone().cpu() * STD_3 + MEAN_3
    x = x.clamp(0.0, 1.0)
    rgb = x.permute(1, 2, 0).numpy()
    return rgb


def plot_training_curves(train_losses, val_maes, output_path):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, train_losses, marker="o")
    ax[0].set_title("Training Loss (MSE)")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE Loss")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, val_maes, marker="o", color="orange")
    ax[1].set_title("Validation MAE (ft)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("MAE (ft)")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter_preds_vs_targets(targets_ft, preds_ft, metrics, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets_ft, preds_ft, alpha=0.5, s=20, c="#4CAF50")

    mn = min(targets_ft.min(), preds_ft.min()) - 0.5
    mx = max(targets_ft.max(), preds_ft.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2)

    ax.set_xlabel("Actual Water Level (ft)", fontsize=12)
    ax.set_ylabel("Predicted Water Level (ft)", fontsize=12)
    ax.set_title(
        f"Triplet RGB Model\n"
        f"MAE: {metrics['mae']:.3f} ft ({metrics['mae']*12:.1f} in), "
        f"R²: {metrics['r2']:.3f}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_sample_visualizations(
    samples,
    elev_mean,
    elev_std,
    output_dir,
    max_samples: int = 4,
):
    """
    samples: list of dicts with:
        ref1, ref2, query: [3,H,W] tensors
        pred, target: normalized (float)
    """
    max_samples = min(max_samples, len(samples))
    for idx in range(max_samples):
        s = samples[idx]

        ref1_rgb = denorm_3ch(s["ref1"])
        ref2_rgb = denorm_3ch(s["ref2"])
        q_rgb = denorm_3ch(s["query"])

        target_ft = s["target"] * elev_std + elev_mean
        pred_ft = s["pred"] * elev_std + elev_mean

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(ref1_rgb)
        axes[0].set_title("Ref1 RGB")
        axes[1].imshow(ref2_rgb)
        axes[1].set_title("Ref2 RGB")
        axes[2].imshow(q_rgb)
        axes[2].set_title(
            f"Query RGB\nTrue: {target_ft:.2f} ft | Pred: {pred_ft:.2f} ft"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(output_dir / f"sample_{idx+1}.png", dpi=150)
        plt.close(fig)


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    print("\n" + "=" * 70)
    print(" TRIPLET MODEL TRAINING (RGB)")
    print("=" * 70)

    # ----------------- datasets -----------------
    print("\n Loading datasets...")
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

    # ----------------- dataloaders -----------------
    batch_size = max(1, config["batch_size"] // 2)

    if DEVICE.type == "cuda":
        num_workers = max(2, os.cpu_count() // 2)
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **loader_kwargs
    )

    # ----------------- model & optimizer -----------------
    print("\n Creating model...")
    model = TripletWaterLevelModel(
        backbone=config["backbone"],
        feature_dim=192,
        hidden_dim=128,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # ----------------- output dir -----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"triplet_rgb_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- training loop -----------------
    print("\n Training...")
    best_val_mae = float("inf")
    train_losses = []
    val_maes = []

    non_blocking = DEVICE.type == "cuda"

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss_sum, train_mae_sum, n_train = 0.0, 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
        )

        for batch in pbar:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
            query = batch["query"].to(DEVICE, non_blocking=non_blocking)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)
            targets = batch["elevation_query"].to(DEVICE, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)
                loss = F.mse_loss(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            train_loss_sum += loss.item() * bs
            train_mae_sum += (
                F.l1_loss(preds, targets, reduction="sum").item()
            )
            n_train += bs

            pbar.set_postfix(
                {
                    "loss": f"{train_loss_sum / n_train:.4f}",
                    "mae_ft": f"{(train_mae_sum / n_train) * train_dataset.elev_std:.3f}",
                }
            )

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        epoch_train_loss = train_loss_sum / n_train
        train_losses.append(epoch_train_loss)

        # ----------------- validation -----------------
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
                ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
                query = batch["query"].to(DEVICE, non_blocking=non_blocking)
                elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
                elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)

                with autocast(enabled=(DEVICE.type == "cuda")):
                    preds = model(ref1, ref2, query, elev1, elev2)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch["elevation_query"].numpy())

        val_preds_ft = (
            np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        )
        val_targets_ft = (
            np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        )

        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        scheduler.step()

        val_mae = val_metrics["mae"]
        val_maes.append(val_mae)

        print(
            f"   Train Loss: {epoch_train_loss:.4f} | "
            f"Val MAE: {val_mae:.3f} ft | Val R²: {val_metrics['r2']:.3f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "elev_mean": train_dataset.elev_mean,
                    "elev_std": train_dataset.elev_std,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
            print("   ✓ Saved best model!")

    # ----------------- save training curves -----------------
    plot_training_curves(
        train_losses,
        val_maes,
        output_dir / "training_curves.png",
    )

    # ----------------- TEST EVALUATION -----------------
    print("\n Evaluating on TEST set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    sample_vis_buffer = []

    with torch.no_grad():
        for batch in test_loader:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
            query = batch["query"].to(DEVICE, non_blocking=non_blocking)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)
            eq = batch["elevation_query"].to(DEVICE, non_blocking=non_blocking)

            with autocast(enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)

            preds_np = preds.cpu().numpy()
            eq_np = eq.cpu().numpy()

            test_preds.extend(preds_np)
            test_targets.extend(eq_np)

            # store a few samples for visualization
            if len(sample_vis_buffer) < 6:
                for i in range(min(ref1.size(0), 6 - len(sample_vis_buffer))):
                    sample_vis_buffer.append(
                        {
                            "ref1": batch["ref1"][i].cpu(),
                            "ref2": batch["ref2"][i].cpu(),
                            "query": batch["query"][i].cpu(),
                            "pred": preds_np[i],
                            "target": eq_np[i],
                        }
                    )

    test_preds_ft = (
        np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    )
    test_targets_ft = (
        np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    )

    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)

    # Scatter: Pred vs Actual
    plot_scatter_preds_vs_targets(
        test_targets_ft,
        test_preds_ft,
        test_metrics,
        output_dir / "test_predictions_scatter.png",
    )

    # Visual samples (Ref1, Ref2, Query with True/Pred)
    save_sample_visualizations(
        sample_vis_buffer,
        train_dataset.elev_mean,
        train_dataset.elev_std,
        output_dir,
        max_samples=4,
    )

    # Save results JSON
    results = {
        "model": "triplet_rgb",
        "test_mae_ft": test_metrics["mae"],
        "test_mae_inches": test_metrics["mae"] * 12.0,
        "test_rmse_ft": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_pearson_r": test_metrics["pearson_r"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(" TRIPLET RGB RESULTS")
    print("=" * 70)
    print(
        f"Test MAE:  {test_metrics['mae']:.3f} ft "
        f"({test_metrics['mae'] * 12:.1f} inches)"
    )
    print(f"Test RMSE: {test_metrics['rmse']:.3f} ft")
    print(f"Test R²:   {test_metrics['r2']:.3f}")
    print(f"Output:    {output_dir}")
    print("Saved:")
    print("  - training_curves.png")
    print("  - test_predictions_scatter.png")
    print("  - sample_1..N.png (RGB triplets + prediction)")
    print("  - results.json")

    return test_metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    config = {
        "train_csv": "data/train_quality.csv",
        "val_csv": "data/val_quality.csv",
        "test_csv": "data/test_quality.csv",
        "backbone": "vit_tiny_patch16_224",
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "min_elev_diff": 0.3,
        "max_triplets": 5000,
    }

    train(config)
