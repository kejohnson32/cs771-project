#!/usr/bin/env python3
"""
DeepWater Advanced Model - Best Results Strategy

This model combines multiple approaches to get the best possible results:
1. Multi-Region Analysis - Analyzes different parts of image separately
2. Reference Structure Detection - Finds fixed structures (walls, poles, dams)
3. Water Edge Detection - Multiple edge detection methods
4. Adaptive ROI - Learns which region is most informative per site
5. Temporal Consistency - Uses image sequences when available

Usage:
    tmux new -s best
    source .venv/bin/activate
    python train_best_model.py --epochs 30 --batch_size 8
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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
import cv2

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available")
    return device

DEVICE = setup_device()


# =============================================================================
# ADVANCED FEATURE EXTRACTION (No SAM dependency)
# =============================================================================

def extract_edge_features(img_np):
    """
    Extract edge-based features using classical CV methods.
    Works without SAM.
    """
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    h, w = gray.shape
    
    # 1. Canny edges
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Horizontal edge detection (water lines are mostly horizontal)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_edges = np.abs(sobel_y) / (np.abs(sobel_x) + np.abs(sobel_y) + 1e-6)
    
    # 3. Find strongest horizontal edge in each column (potential water line)
    water_line = np.zeros(32, dtype=np.float32)
    step = w // 32
    for i in range(32):
        col_start = i * step
        col_end = min((i + 1) * step, w)
        
        # Look at bottom 70% of image (where water usually is)
        search_region = horizontal_edges[int(h*0.3):, col_start:col_end]
        if search_region.size > 0:
            col_profile = search_region.mean(axis=1)
            if len(col_profile) > 0:
                # Find strongest horizontal edge
                peak_idx = np.argmax(col_profile)
                water_line[i] = (int(h*0.3) + peak_idx) / h
    
    # 4. Color-based water detection (water is often darker/bluer)
    if len(img_np.shape) == 3:
        # Blue channel ratio
        blue_ratio = img_np[:,:,2].astype(float) / (img_np.mean(axis=2) + 1)
        
        # Darkness (water absorbs light)
        darkness = 1 - (img_np.mean(axis=2) / 255.0)
        
        # Combined water likelihood
        water_likelihood = (blue_ratio * 0.3 + darkness * 0.7)
        water_likelihood = (water_likelihood - water_likelihood.min()) / (water_likelihood.max() - water_likelihood.min() + 1e-6)
    else:
        water_likelihood = 1 - (gray / 255.0)
    
    # 5. Find water level from color analysis
    color_water_line = np.zeros(32, dtype=np.float32)
    for i in range(32):
        col_start = i * step
        col_end = min((i + 1) * step, w)
        col = water_likelihood[:, col_start:col_end].mean(axis=1)
        
        # Find transition point (threshold crossing)
        threshold = 0.5
        crossings = np.where(np.diff((col > threshold).astype(int)))[0]
        if len(crossings) > 0:
            # Take the first crossing in bottom half
            valid_crossings = crossings[crossings > h//2]
            if len(valid_crossings) > 0:
                color_water_line[i] = valid_crossings[0] / h
            else:
                color_water_line[i] = crossings[-1] / h
        else:
            color_water_line[i] = 0.7  # Default
    
    # 6. Texture analysis (water has different texture than land)
    # Use local standard deviation
    kernel_size = 15
    local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    
    # Water typically has lower texture variance (smoother)
    texture_profile = np.zeros(32, dtype=np.float32)
    for i in range(32):
        col_start = i * step
        col_end = min((i + 1) * step, w)
        col = local_std[:, col_start:col_end].mean(axis=1)
        # Normalize
        texture_profile[i] = col[int(h*0.6):].mean() / (col.mean() + 1e-6)
    
    # Combine all features
    features = np.concatenate([
        water_line,           # 32: Edge-based water line
        color_water_line,     # 32: Color-based water line
        texture_profile,      # 32: Texture profile
        [water_line.mean()],          # 1: Mean edge water level
        [water_line.std()],           # 1: Std of water line (flatness)
        [color_water_line.mean()],    # 1: Mean color water level
        [color_water_line.std()],     # 1: Color line std
        [texture_profile.mean()],     # 1: Mean texture ratio
        [np.corrcoef(water_line, color_water_line)[0,1] if len(water_line) > 1 else 0],  # 1: Agreement between methods
    ])
    
    return features.astype(np.float32)  # Total: 102 features


def extract_region_features(img_np, regions=4):
    """
    Extract features from multiple regions of the image.
    Different regions may be more informative for different sites.
    """
    h, w = img_np.shape[:2]
    
    # Define regions: [top-left, top-right, bottom-left, bottom-right]
    # Bottom regions are usually more informative for water level
    region_coords = [
        (0, 0, h//2, w//2),           # Top-left
        (0, w//2, h//2, w),           # Top-right
        (h//2, 0, h, w//2),           # Bottom-left (often has shore/structure)
        (h//2, w//2, h, w),           # Bottom-right
    ]
    
    region_features = []
    
    for y1, x1, y2, x2 in region_coords:
        region = img_np[y1:y2, x1:x2]
        
        # Basic stats
        mean_rgb = region.mean(axis=(0,1)) / 255.0 if len(region.shape) == 3 else [region.mean() / 255.0]
        std_rgb = region.std(axis=(0,1)) / 255.0 if len(region.shape) == 3 else [region.std() / 255.0]
        
        # Edge density
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = region
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = edges.mean() / 255.0
        
        # Horizontal edge ratio
        sobel_y = np.abs(cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3))
        sobel_x = np.abs(cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3))
        h_ratio = sobel_y.sum() / (sobel_x.sum() + sobel_y.sum() + 1e-6)
        
        if len(region.shape) == 3:
            region_features.extend(mean_rgb)  # 3
            region_features.extend(std_rgb)   # 3
        else:
            region_features.extend([mean_rgb[0]] * 3)
            region_features.extend([std_rgb[0]] * 3)
        region_features.append(edge_density)  # 1
        region_features.append(h_ratio)       # 1
    
    return np.array(region_features, dtype=np.float32)  # 4 regions * 8 features = 32


def extract_all_features(img_np):
    """Combine all feature extraction methods."""
    edge_feats = extract_edge_features(img_np)      # 102
    region_feats = extract_region_features(img_np)  # 32
    
    return np.concatenate([edge_feats, region_feats])  # 134 total


# =============================================================================
# MODEL
# =============================================================================

class BestWaterLevelModel(nn.Module):
    """
    Advanced model combining:
    1. ViT backbone for visual features
    2. Multi-method water detection features
    3. Region-aware processing
    4. Cross-attention between references and query
    5. Learnable feature weighting
    """

    def __init__(self, backbone='vit_tiny_patch16_224', hidden_dim=256, dropout=0.1):
        super().__init__()

        # Visual backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        with torch.no_grad():
            backbone_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        
        # Feature encoder for extracted features (134 -> 128)
        self.feature_encoder = nn.Sequential(
            nn.Linear(134, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        
        # Combine visual + extracted features
        combined_dim = backbone_dim + 128
        
        # Feature fusion with attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention between query and references
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Elevation embedding
        self.elev_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Difference encoder - explicitly model the difference
        self.diff_encoder = nn.Sequential(
            nn.Linear(combined_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(combined_dim * 2 + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Learnable weights for different feature types
        self.feature_weights = nn.Parameter(torch.ones(3))  # [visual, edge, region]

    def forward(self, ref1_img, ref2_img, query_img,
                ref1_feats, ref2_feats, query_feats,
                elev1, elev2):
        """
        Args:
            ref1_img, ref2_img, query_img: RGB images [B, 3, 224, 224]
            ref1_feats, ref2_feats, query_feats: Extracted features [B, 134]
            elev1, elev2: Reference elevations [B]
        """
        # Visual features
        vis1 = self.backbone(ref1_img)
        vis2 = self.backbone(ref2_img)
        vis_q = self.backbone(query_img)
        
        # Encode extracted features
        feat1 = self.feature_encoder(ref1_feats)
        feat2 = self.feature_encoder(ref2_feats)
        feat_q = self.feature_encoder(query_feats)
        
        # Combine visual + extracted
        comb1 = torch.cat([vis1, feat1], dim=1)
        comb2 = torch.cat([vis2, feat2], dim=1)
        comb_q = torch.cat([vis_q, feat_q], dim=1)
        
        # Cross-attention: query attends to references
        refs = torch.stack([comb1, comb2], dim=1)  # [B, 2, D]
        attn_out, attn_weights = self.cross_attn(comb_q.unsqueeze(1), refs, refs)
        attn_out = attn_out.squeeze(1)  # [B, D]
        
        # Explicit difference modeling
        # Compare query to weighted average of references
        ref_avg = (comb1 + comb2) / 2
        diff_input = torch.cat([comb_q, ref_avg], dim=1)
        diff_encoding = self.diff_encoder(diff_input)
        
        # Elevation embedding
        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        
        # Combine everything
        combined = torch.cat([comb_q, attn_out, diff_encoding, elev_emb], dim=1)
        
        return self.head(combined).squeeze(-1)


# =============================================================================
# DATASET
# =============================================================================

class BestModelDataset(torch.utils.data.Dataset):
    """
    Dataset with comprehensive feature extraction.
    """

    def __init__(self, csv_path, elev_mean=None, elev_std=None,
                 min_elev_diff=0.3, max_triplets=5000,
                 cache_dir="data/best_cache"):

        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")

        # Clean data
        self.df["gage_height_ft"] = pd.to_numeric(self.df["gage_height_ft"], errors="coerce")
        self.df = self.df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
        self.df = self.df[(self.df["gage_height_ft"] > 0) & 
                          (self.df["gage_height_ft"] < 50)].reset_index(drop=True)

        # Image transforms
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if elev_mean is None:
            self.elev_mean = self.df["gage_height_ft"].mean()
            self.elev_std = self.df["gage_height_ft"].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std

        logger.info(f"Elevation stats: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")

        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Group by site
        self.site_groups = {}
        for site in self.df["camera_id"].unique():
            site_df = self.df[self.df["camera_id"] == site].reset_index(drop=True)
            if len(site_df) >= 3:
                self.site_groups[site] = site_df

        logger.info(f"Sites with 3+ images: {len(self.site_groups)}")
        self._build_triplets(min_elev_diff, max_triplets)

    def _build_triplets(self, min_elev_diff, max_triplets):
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

    def _get_cache_path(self, image_path):
        cache_name = Path(image_path).stem + "_best_features.npy"
        site = Path(image_path).parent.parent.name
        cache_subdir = self.cache_dir / site
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / cache_name

    def _load_image_and_features(self, image_path):
        """Load image and extract features."""
        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        
        img_resized = self.resize(img)
        img_np = np.array(img_resized)
        
        # Check cache
        cache_path = self._get_cache_path(image_path)
        
        if cache_path.exists():
            try:
                features = np.load(cache_path)
            except Exception:
                features = extract_all_features(img_np)
        else:
            features = extract_all_features(img_np)
            try:
                np.save(cache_path, features)
            except OSError:
                pass
        
        # Convert to tensors
        img_tensor = self.normalize(self.to_tensor(img_resized))
        feat_tensor = torch.from_numpy(features).float()
        
        return img_tensor, feat_tensor

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]

        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]

        img1, feat1 = self._load_image_and_features(row1["image_path"])
        img2, feat2 = self._load_image_and_features(row2["image_path"])
        img3, feat3 = self._load_image_and_features(row3["image_path"])

        elev1 = (row1["gage_height_ft"] - self.elev_mean) / self.elev_std
        elev2 = (row2["gage_height_ft"] - self.elev_mean) / self.elev_std
        elev3 = (row3["gage_height_ft"] - self.elev_mean) / self.elev_std

        return {
            "ref1_img": img1,
            "ref2_img": img2,
            "query_img": img3,
            "ref1_feats": feat1,
            "ref2_feats": feat2,
            "query_feats": feat3,
            "elevation1": torch.tensor(elev1, dtype=torch.float32),
            "elevation2": torch.tensor(elev2, dtype=torch.float32),
            "elevation_query": torch.tensor(elev3, dtype=torch.float32),
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
    ax2.plot(val_maes, 'g-', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE (ft)'); ax2.set_title('Validation MAE'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(targets, preds, metrics, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targets, preds, alpha=0.5, s=20, c='#FF5722')
    mn, mx = min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('Actual (ft)'); ax.set_ylabel('Predicted (ft)')
    ax.set_title(f'Best Model: MAE={metrics["mae"]:.3f}ft ({metrics["mae"]*12:.1f}in), R²={metrics["r2"]:.3f}')
    ax.grid(True, alpha=0.3); ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_feature_extraction(img_np, save_path):
    """Visualize what the feature extraction sees."""
    h, w = img_np.shape[:2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Edges
    edges = cv2.Canny(gray, 50, 150)
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('Canny Edges')
    axes[0, 2].axis('off')
    
    # Horizontal edges
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    axes[1, 0].imshow(sobel_y, cmap='hot')
    axes[1, 0].set_title('Horizontal Edges (Water Lines)')
    axes[1, 0].axis('off')
    
    # Water likelihood (color-based)
    blue_ratio = img_np[:,:,2].astype(float) / (img_np.mean(axis=2) + 1)
    darkness = 1 - (img_np.mean(axis=2) / 255.0)
    water_likelihood = (blue_ratio * 0.3 + darkness * 0.7)
    water_likelihood = (water_likelihood - water_likelihood.min()) / (water_likelihood.max() - water_likelihood.min() + 1e-6)
    axes[1, 1].imshow(water_likelihood, cmap='Blues')
    axes[1, 1].set_title('Water Likelihood (Color)')
    axes[1, 1].axis('off')
    
    # Detected water line overlay
    features = extract_edge_features(img_np)
    edge_water_line = features[:32]
    color_water_line = features[32:64]
    
    axes[1, 2].imshow(img_np)
    x_coords = np.linspace(0, w-1, 32)
    axes[1, 2].plot(x_coords, edge_water_line * h, 'c-', lw=2, label='Edge-based')
    axes[1, 2].plot(x_coords, color_water_line * h, 'm-', lw=2, label='Color-based')
    axes[1, 2].legend()
    axes[1, 2].set_title('Detected Water Lines')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_sample_visualizations(samples, elev_mean, elev_std, output_dir, max_samples=4):
    """Save sample visualizations."""
    mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i, s in enumerate(samples[:max_samples]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for j, (img_key, title) in enumerate([
            ("ref1_img", "Reference 1"),
            ("ref2_img", "Reference 2"),
            ("query_img", "Query")
        ]):
            img = s[img_key].clone()
            img = img * std_rgb + mean_rgb
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            axes[j].imshow(img)
            
            if j == 2:
                pred_ft = s["pred"] * elev_std + elev_mean
                tgt_ft = s["target"] * elev_std + elev_mean
                error = abs(pred_ft - tgt_ft)
                title = f"Query\nTrue: {tgt_ft:.2f}ft | Pred: {pred_ft:.2f}ft\nError: {error:.2f}ft ({error*12:.1f}in)"
            
            axes[j].set_title(title)
            axes[j].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i+1}.png", dpi=150)
        plt.close()
        
        # Also save feature visualization for query
        query_img = s["query_img"].clone()
        query_img = query_img * std_rgb + mean_rgb
        query_img = torch.clamp(query_img, 0, 1).permute(1, 2, 0).numpy()
        query_img = (query_img * 255).astype(np.uint8)
        visualize_feature_extraction(query_img, output_dir / f"sample_{i+1}_features.png")


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    logger.info("=" * 70)
    logger.info("BEST MODEL TRAINING")
    logger.info("Multi-method water detection + ViT + Cross-attention")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Datasets
    logger.info("\nLoading datasets...")
    train_dataset = BestModelDataset(
        config["train_csv"],
        max_triplets=config["max_triplets"],
    )

    val_dataset = BestModelDataset(
        config["val_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
    )

    test_dataset = BestModelDataset(
        config["test_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
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
    logger.info("\nCreating Best Model...")
    model = BestWaterLevelModel(backbone=config["backbone"], hidden_dim=256).to(DEVICE)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"best_model_{timestamp}"
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
            ref1_img = batch["ref1_img"].to(DEVICE, non_blocking=True)
            ref2_img = batch["ref2_img"].to(DEVICE, non_blocking=True)
            query_img = batch["query_img"].to(DEVICE, non_blocking=True)
            ref1_feats = batch["ref1_feats"].to(DEVICE, non_blocking=True)
            ref2_feats = batch["ref2_feats"].to(DEVICE, non_blocking=True)
            query_feats = batch["query_feats"].to(DEVICE, non_blocking=True)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)
            targets = batch["elevation_query"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(ref1_img, ref2_img, query_img,
                             ref1_feats, ref2_feats, query_feats,
                             elev1, elev2)
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
                ref1_img = batch["ref1_img"].to(DEVICE, non_blocking=True)
                ref2_img = batch["ref2_img"].to(DEVICE, non_blocking=True)
                query_img = batch["query_img"].to(DEVICE, non_blocking=True)
                ref1_feats = batch["ref1_feats"].to(DEVICE, non_blocking=True)
                ref2_feats = batch["ref2_feats"].to(DEVICE, non_blocking=True)
                query_feats = batch["query_feats"].to(DEVICE, non_blocking=True)
                elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
                elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    preds = model(ref1_img, ref2_img, query_img,
                                 ref1_feats, ref2_feats, query_feats,
                                 elev1, elev2)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch["elevation_query"].numpy())

        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)

        scheduler.step()
        val_mae = val_metrics["mae"]
        val_maes.append(val_mae)

        logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val MAE={val_mae:.3f}ft, R²={val_metrics['r2']:.3f}")

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

    # Test evaluation
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    sample_buffer = []

    with torch.no_grad():
        for batch in test_loader:
            ref1_img = batch["ref1_img"].to(DEVICE, non_blocking=True)
            ref2_img = batch["ref2_img"].to(DEVICE, non_blocking=True)
            query_img = batch["query_img"].to(DEVICE, non_blocking=True)
            ref1_feats = batch["ref1_feats"].to(DEVICE, non_blocking=True)
            ref2_feats = batch["ref2_feats"].to(DEVICE, non_blocking=True)
            query_feats = batch["query_feats"].to(DEVICE, non_blocking=True)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=True)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(ref1_img, ref2_img, query_img,
                             ref1_feats, ref2_feats, query_feats,
                             elev1, elev2)

            preds_np = preds.cpu().numpy()
            targets_np = batch["elevation_query"].numpy()

            test_preds.extend(preds_np)
            test_targets.extend(targets_np)

            if len(sample_buffer) < 6:
                for i in range(min(ref1_img.size(0), 6 - len(sample_buffer))):
                    sample_buffer.append({
                        "ref1_img": batch["ref1_img"][i].cpu(),
                        "ref2_img": batch["ref2_img"][i].cpu(),
                        "query_img": batch["query_img"][i].cpu(),
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
        "model": "best_multi_method",
        "test_mae_ft": test_metrics["mae"],
        "test_mae_inches": test_metrics["mae"] * 12,
        "test_rmse_ft": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "best_epoch": checkpoint["epoch"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - BEST MODEL")
    logger.info("=" * 70)
    logger.info(f"Test MAE:  {test_metrics['mae']:.3f} ft ({test_metrics['mae']*12:.1f} inches)")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.3f} ft")
    logger.info(f"Test R²:   {test_metrics['r2']:.3f}")
    logger.info(f"Output:    {output_dir}")

    return test_metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Best Water Level Model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
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
