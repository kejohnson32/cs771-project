#!/usr/bin/env python3
"""
Train Siamese model for water level estimation (RGB only).

Usage:
    python train_siamese_5k.py --epochs 30
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import timm
from PIL import Image
import torchvision.transforms as T

# =============================================================================
# DEVICE SETUP
# =============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f" Using CUDA: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f" Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print(f" Using CPU")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
# =============================================================================
# MODEL
# =============================================================================

class SiameseWaterLevelModel(nn.Module):
    """Siamese network for water level estimation."""
    
    def __init__(self, backbone='vit_tiny_patch16_224', feature_dim=192, 
                 hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Shared backbone (ViT)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        # Get feature dimension from backbone
        with torch.no_grad():
            backbone_dim = self.backbone.num_features
        
        # Elevation embedding
        self.elev_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Fusion and prediction head
        # Input: concat(feat1, feat2, elev_embed) = backbone_dim * 2 + 32
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, img1, img2, elev1):
        """
        Args:
            img1: Reference image [B, 3, 224, 224]
            img2: Query image [B, 3, 224, 224]
            elev1: Reference elevation (normalized) [B]
        Returns:
            Predicted elevation for img2 (normalized) [B]
        """
        # Extract features
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        
        # Embed elevation
        elev_emb = self.elev_embed(elev1.unsqueeze(-1))
        
        # Fuse and predict
        combined = torch.cat([feat1, feat2, elev_emb], dim=1)
        pred = self.head(combined).squeeze(-1)
        
        return pred


# =============================================================================
# DATASET
# =============================================================================

class SameSitePairDataset(torch.utils.data.Dataset):
    """Dataset that creates pairs ONLY from the same camera site."""
    
    def __init__(self, csv_path, elev_mean=None, elev_std=None, 
                 min_elev_diff=0.2, max_pairs_per_site=2000):
        
        self.df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Clean data
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & 
                          (self.df['gage_height_ft'] < 50)].reset_index(drop=True)
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Normalization stats
        if elev_mean is None:
            self.elev_mean = self.df['gage_height_ft'].mean()
            self.elev_std = self.df['gage_height_ft'].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std
        
        print(f"Elevation normalization: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")
        
        # Group by site
        self.site_groups = {}
        for site in self.df['camera_id'].unique():
            site_df = self.df[self.df['camera_id'] == site].reset_index(drop=True)
            if len(site_df) >= 2:
                self.site_groups[site] = site_df
        
        print(f"Sites with 2+ images: {len(self.site_groups)}")
        
        # Build same-site pairs
        self.pairs = []
        for site, site_df in self.site_groups.items():
            site_pairs = []
            n = len(site_df)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        elev_diff = abs(site_df.iloc[i]['gage_height_ft'] - 
                                       site_df.iloc[j]['gage_height_ft'])
                        if elev_diff >= min_elev_diff:
                            site_pairs.append((site, i, j))
            
            # Limit pairs per site
            if len(site_pairs) > max_pairs_per_site:
                np.random.shuffle(site_pairs)
                site_pairs = site_pairs[:max_pairs_per_site]
            
            self.pairs.extend(site_pairs)
            print(f"  {site}: {len(site_pairs)} pairs")
        
        print(f"Total same-site pairs: {len(self.pairs)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        site, i, j = self.pairs[idx]
        site_df = self.site_groups[site]
        
        row1 = site_df.iloc[i]
        row2 = site_df.iloc[j]
        
        img1 = Image.open(row1['image_path']).convert('RGB')
        img2 = Image.open(row2['image_path']).convert('RGB')
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        elev1 = torch.tensor((row1['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32)
        elev2 = torch.tensor((row2['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32)
        
        return {
            'image1': img1,
            'image2': img2,
            'elevation1': elev1,
            'elevation2': elev2,
            'site': site,
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, targets):
    """Compute regression metrics."""
    preds = np.array(preds)
    targets = np.array(targets)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    if len(preds) > 1:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': corr}


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    print('\n' + '='*70)
    print(' SIAMESE MODEL TRAINING (RGB)')
    print('='*70)
    
    # Create datasets
    print('\n Loading datasets...')
    train_dataset = SameSitePairDataset(
        config['train_csv'],
        min_elev_diff=config['min_elev_diff'],
        max_pairs_per_site=config['max_pairs_per_site']
    )
    
    
    test_dataset = SameSitePairDataset(
        config['test_csv'],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        min_elev_diff=config['min_elev_diff'],
        max_pairs_per_site=config['max_pairs_per_site']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    
    # Create model
    print('\n Creating model...')
    model = SiameseWaterLevelModel(
        backbone=config['backbone'],
        feature_dim=192,
        hidden_dim=128,
        dropout=0.1
    )
    model.to(DEVICE, memory_format=torch.channels_last)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'   Parameters: {n_params/1e6:.2f}M')
    

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f'siamese_rgb_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print('\n Training...')
    best_val_mae = float('inf')
    history = []
    # Test evaluation
    print('\n Evaluating on TEST set (unseen sites)...')
    checkpoint = torch.load('outputs/test_siamese_rgb_unseen/best_model_rgb.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            img1 = batch['image1'].to(DEVICE, non_blocking=True)
            img2 = batch['image2'].to(DEVICE, non_blocking=True)
            elev1 = batch['elevation1'].to(DEVICE, non_blocking=True)
            
            preds = model(img1, img2, elev1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(batch['elevation2'].numpy())
    
    test_preds_ft = np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    test_targets_ft = np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    
    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(test_targets_ft, test_preds_ft, alpha=0.5, s=20, c='#2196F3')
    
    mn = min(test_targets_ft.min(), test_preds_ft.min()) - 0.5
    mx = max(test_targets_ft.max(), test_preds_ft.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual Water Level (ft)', fontsize=12)
    ax.set_ylabel('Predicted Water Level (ft)', fontsize=12)
    ax.set_title(f'Siamese RGB Model\n'
                 f'MAE: {test_metrics["mae"]:.3f} ft ({test_metrics["mae"]*12:.1f} in), '
                 f'R²: {test_metrics["r2"]:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    
    plt.tight_layout()
    plt.savefig('outputs/test_siamese_rgb_unseen/test_predictions.png', dpi=150)
    
    # Save results
    import json
    results = {
        'model': 'siamese_rgb',
        'test_mae_ft': test_metrics['mae'],
        'test_mae_inches': test_metrics['mae'] * 12,
        'test_rmse_ft': test_metrics['rmse'],
        'test_r2': test_metrics['r2'],
        'test_pearson_r': test_metrics['pearson_r'],
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'config': config,
    }
    
    with open('outputs/test_siamese_rgb_unseen/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print('\n' + '='*70)
    print(' SIAMESE RGB RESULTS')
    print('='*70)
    print(f'Test MAE:  {test_metrics["mae"]:.3f} ft ({test_metrics["mae"]*12:.1f} inches)')
    print(f'Test RMSE: {test_metrics["rmse"]:.3f} ft')
    print(f'Test R²:   {test_metrics["r2"]:.3f}')
    print(f'Output:    {output_dir}')
    
    return test_metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    config = {
        'train_csv': 'data/train_quality.csv',
        'test_csv': 'data/test_quality.csv',
        'backbone': 'vit_tiny_patch16_224',
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'min_elev_diff': 0.2,
        'max_pairs_per_site': 2000,
    }
    
    train(config)
