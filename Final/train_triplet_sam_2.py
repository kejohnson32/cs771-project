#!/usr/bin/env python3
"""
Train Triplet model with SAM2 water masks (RGB + Mask = 4 channels).
Optimized for NVIDIA T4 GPU on Google Cloud Platform.

Usage:
    # In tmux session on GCP:
    python train_triplet_sam.py --epochs 50 --batch_size 12 --lr 1e-4

    # First run will generate SAM masks (cached for future runs)
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# DEVICE SETUP
# =============================================================================

def setup_device():
    """Configure device with T4 optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    return device

DEVICE = setup_device()

# Normalization constants for 4-channel (RGB + mask)
MEAN_4CH = [0.485, 0.456, 0.406, 0.5]
STD_4CH = [0.229, 0.224, 0.225, 0.5]

# =============================================================================
# SAM2 SETUP
# =============================================================================

SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "sam2/checkpoints/sam2.1_hiera_small.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_sam2_predictor = None

def get_sam2_predictor():
    """Lazy load SAM2 predictor."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info(f"Loading SAM2 from {SAM2_CHECKPOINT}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load SAM2: {e}")
        logger.warning("Using dummy masks (all ones)")
        _sam2_predictor = "dummy"

    return _sam2_predictor


def generate_water_mask(image_np, predictor):
    """Generate water segmentation mask using SAM2."""
    if predictor == "dummy":
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    try:
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        
        # Use center-bottom point as prompt (water usually in lower part)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, int(h * 0.7)]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(np.float32)
    except Exception as e:
        logger.warning(f"SAM2 mask generation failed: {e}")
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)


# =============================================================================
# MODEL
# =============================================================================

class TripletSAMModel(nn.Module):
    """
    Triplet network with 4-channel input (RGB + water mask).
    
    The mask channel helps the model focus on water regions.
    """

    def __init__(self, backbone='vit_tiny_patch16_224', hidden_dim=128, dropout=0.1):
        super().__init__()

        # Create 4-channel backbone
        base_model = timm.create_model(backbone, pretrained=True, num_classes=0)

        if hasattr(base_model, "patch_embed"):
            old_proj = base_model.patch_embed.proj
            new_proj = nn.Conv2d(
                4, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
            )
            with torch.no_grad():
                new_proj.weight[:, :3] = old_proj.weight
                new_proj.weight[:, 3:] = old_proj.weight[:, :1]  # Init from R channel
                new_proj.bias = old_proj.bias
            base_model.patch_embed.proj = new_proj

        self.backbone = base_model

        with torch.no_grad():
            backbone_dim = self.backbone(torch.randn(1, 4, 224, 224)).shape[1]

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.elev_embed = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
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
        attn_out, _ = self.cross_attn(feat_q.unsqueeze(1), refs, refs)

        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        combined = torch.cat([feat_q, attn_out.squeeze(1), elev_emb], dim=1)
        
        return self.head(combined).squeeze(-1)


# =============================================================================
# DATASET
# =============================================================================

def safe_load_image(path, size=(224, 224)):
    """Safely load image."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return Image.new("RGB", size, (0, 0, 0))


class SameSiteTripletDatasetSAM(torch.utils.data.Dataset):
    """
    Triplet dataset with SAM water masks as 4th channel.
    Masks are cached to disk for faster subsequent runs.
    """

    def __init__(self, csv_path, elev_mean=None, elev_std=None,
                 min_elev_diff=0.3, max_triplets=5000,
                 cache_masks=True, mask_cache_dir="data/mask_cache"):

        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")

        # Clean data
        self.df["gage_height_ft"] = pd.to_numeric(self.df["gage_height_ft"], errors="coerce")
        self.df = self.df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
        self.df = self.df[(self.df["gage_height_ft"] > 0) & 
                          (self.df["gage_height_ft"] < 50)].reset_index(drop=True)

        # Transforms
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=MEAN_4CH, std=STD_4CH)

        if elev_mean is None:
            self.elev_mean = self.df["gage_height_ft"].mean()
            self.elev_std = self.df["gage_height_ft"].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std

        logger.info(f"Elevation stats: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")

        # Mask caching
        self.cache_masks = cache_masks
        self.mask_cache_dir = Path(mask_cache_dir)
        if cache_masks:
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

        self.sam_predictor = get_sam2_predictor()

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
        """Build same-site triplets."""
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
            if abs(site_df.iloc[i]["gage_height_ft"] - site_df.iloc[j]["gage_height_ft"]) >= min_elev_diff:
                self.triplets.append((site, i, j, k))

            attempts += 1

        logger.info(f"Created {len(self.triplets)} triplets")

    def _get_mask_cache_path(self, image_path):
        """Get cache path for mask."""
        cache_name = Path(image_path).stem + "_mask.npy"
        site = Path(image_path).parent.parent.name
        cache_dir = self.mask_cache_dir / site
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_name

    def _load_image_with_mask(self, image_path):
        """Load image and generate/load cached mask."""
        img = safe_load_image(image_path)
        img_resized = self.resize(img)
        img_np = np.array(img_resized)

        # Check cache
        if self.cache_masks:
            cache_path = self._get_mask_cache_path(image_path)
            if cache_path.exists():
                try:
                    mask = np.load(cache_path)
                except Exception:
                    mask = generate_water_mask(img_np, self.sam_predictor)
                    mask = np.array(Image.fromarray(
                        (mask * 255).astype(np.uint8)
                    ).resize((224, 224))) / 255.0
            else:
                mask = generate_water_mask(img_np, self.sam_predictor)
                mask = np.array(Image.fromarray(
                    (mask * 255).astype(np.uint8)
                ).resize((224, 224))) / 255.0
                # Try to save cache, but don't fail if disk is full
                try:
                    np.save(cache_path, mask.astype(np.float32))
                except OSError:
                    pass  # Silently skip if disk is full
        else:
            mask = generate_water_mask(img_np, self.sam_predictor)
            mask = np.array(Image.fromarray(
                (mask * 255).astype(np.uint8)
            ).resize((224, 224))) / 255.0

        # Create 4-channel tensor
        img_tensor = self.to_tensor(img_resized)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        img_4ch = torch.cat([img_tensor, mask_tensor], dim=0)
        
        return self.normalize(img_4ch)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]

        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]

        return {
            "ref1": self._load_image_with_mask(row1["image_path"]),
            "ref2": self._load_image_with_mask(row2["image_path"]),
            "query": self._load_image_with_mask(row3["image_path"]),
            "elevation1": torch.tensor(
                (row1["gage_height_ft"] - self.elev_mean) / self.elev_std, dtype=torch.float32
            ),
            "elevation2": torch.tensor(
                (row2["gage_height_ft"] - self.elev_mean) / self.elev_std, dtype=torch.float32
            ),
            "elevation_query": torch.tensor(
                (row3["gage_height_ft"] - self.elev_mean) / self.elev_std, dtype=torch.float32
            ),
        }


# =============================================================================
# METRICS AND VISUALIZATION
# =============================================================================

def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    pearson_r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r)}


def plot_training_curves(train_losses, val_maes, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, 'b-', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
    ax2.plot(val_maes, 'r-', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE (ft)'); ax2.set_title('Validation MAE'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(targets, preds, metrics, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targets, preds, alpha=0.5, s=20, c='#9C27B0')
    mn, mx = min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('Actual (ft)'); ax.set_ylabel('Predicted (ft)')
    ax.set_title(f'Triplet+SAM: MAE={metrics["mae"]:.3f}ft, R2={metrics["r2"]:.3f}')
    ax.grid(True, alpha=0.3); ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def denorm_4ch(tensor):
    """Denormalize 4-channel tensor, return RGB."""
    mean = torch.tensor(MEAN_4CH[:3]).view(3, 1, 1)
    std = torch.tensor(STD_4CH[:3]).view(3, 1, 1)
    rgb = tensor[:3].clone()
    rgb = rgb * std + mean
    return torch.clamp(rgb, 0, 1).permute(1, 2, 0).numpy()


def save_sample_visualizations(samples, elev_mean, elev_std, output_dir, max_samples=4):
    for i, s in enumerate(samples[:max_samples]):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(denorm_4ch(s["ref1"]))
        axes[0].set_title("Ref 1"); axes[0].axis("off")
        
        axes[1].imshow(denorm_4ch(s["ref2"]))
        axes[1].set_title("Ref 2"); axes[1].axis("off")
        
        axes[2].imshow(denorm_4ch(s["query"]))
        pred_ft = s["pred"] * elev_std + elev_mean
        tgt_ft = s["target"] * elev_std + elev_mean
        axes[2].set_title(f"Query\nTrue:{tgt_ft:.2f}ft Pred:{pred_ft:.2f}ft"); axes[2].axis("off")
        
        # Show mask
        mask = s["query"][3].numpy()
        axes[3].imshow(mask, cmap='gray')
        axes[3].set_title("Water Mask"); axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i+1}.png", dpi=150)
        plt.close()


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    logger.info("=" * 70)
    logger.info("TRIPLET + SAM MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Datasets
    logger.info("\nLoading datasets...")
    train_dataset = SameSiteTripletDatasetSAM(
        config["train_csv"],
        max_triplets=config["max_triplets"],
        cache_masks=True,
    )

    val_dataset = SameSiteTripletDatasetSAM(
        config["val_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
        cache_masks=True,
    )

    test_dataset = SameSiteTripletDatasetSAM(
        config["test_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
        cache_masks=True,
    )

    # DataLoaders
    num_workers = 4 if DEVICE.type == "cuda" else 0
    batch_size = max(1, config["batch_size"] // 2)
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Model
    logger.info("\nCreating model...")
    model = TripletSAMModel(backbone=config["backbone"]).to(DEVICE)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"triplet_sam_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    logger.info("\nStarting training...")
    best_val_mae = float("inf")
    train_losses, val_maes = [], []

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss_sum, n_train = 0.0, 0

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

        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)

        scheduler.step()
        val_mae = val_metrics["mae"]
        val_maes.append(val_mae)

        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val MAE={val_mae:.3f}ft, R2={val_metrics['r2']:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model_state_dict": model.state_dict(),
                "elev_mean": train_dataset.elev_mean,
                "elev_std": train_dataset.elev_std,
                "epoch": epoch,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> Saved best model (MAE: {best_val_mae:.3f}ft)")

    # Save curves
    plot_training_curves(train_losses, val_maes, output_dir / "training_curves.png")

    # Test
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
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

            if len(sample_buffer) < 6:
                for i in range(min(ref1.size(0), 6 - len(sample_buffer))):
                    sample_buffer.append({
                        "ref1": batch["ref1"][i].cpu(),
                        "ref2": batch["ref2"][i].cpu(),
                        "query": batch["query"][i].cpu(),
                        "pred": preds_np[i],
                        "target": targets_np[i],
                    })

    test_preds_ft = np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    test_targets_ft = np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)

    # Visualizations
    plot_predictions(test_targets_ft, test_preds_ft, test_metrics, output_dir / "test_predictions.png")
    save_sample_visualizations(sample_buffer, train_dataset.elev_mean, train_dataset.elev_std, output_dir)

    # Results
    results = {
        "model": "triplet_sam",
        "test_mae_ft": test_metrics["mae"],
        "test_mae_inches": test_metrics["mae"] * 12,
        "test_rmse_ft": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "best_epoch": checkpoint["epoch"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - TRIPLET + SAM")
    logger.info("=" * 70)
    logger.info(f"Test MAE:  {test_metrics['mae']:.3f} ft ({test_metrics['mae']*12:.1f} inches)")
    logger.info(f"Test R2:   {test_metrics['r2']:.3f}")
    logger.info(f"Output:    {output_dir}")

    return test_metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Triplet+SAM model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--train_csv", type=str, default="data/train_quality.csv")
    parser.add_argument("--val_csv", type=str, default="data/val_quality.csv")
    parser.add_argument("--test_csv", type=str, default="data/test_quality.csv")
    parser.add_argument("--max_triplets", type=int, default=5000)
    args = parser.parse_args()

    train({
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "backbone": args.backbone,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "max_triplets": args.max_triplets,
    })
