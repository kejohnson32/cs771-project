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
    print(f" Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f" Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print(f" Using CPU")

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
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]
        
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
        
        elev1 = (row1['gage_height_ft'] - self.elev_mean) / self.elev_std
        elev2 = (row2['gage_height_ft'] - self.elev_mean) / self.elev_std
        
        return {
            'image1': img1,
            'image2': img2,
            'elevation1': torch.tensor(elev1, dtype=torch.float32),
            'elevation2': torch.tensor(elev2, dtype=torch.float32),
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
    
    val_dataset = SameSitePairDataset(
        config['val_csv'],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
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
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    
    # Create model
    print('\n Creating model...')
    model = SiameseWaterLevelModel(
        backbone=config['backbone'],
        feature_dim=192,
        hidden_dim=128,
        dropout=0.1
    )
    model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'   Parameters: {n_params/1e6:.2f}M')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f'siamese_rgb_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print('\n Training...')
    best_val_mae = float('inf')
    history = []
    
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        train_loss, train_mae, n = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        for batch in pbar:
            img1 = batch['image1'].to(DEVICE)
            img2 = batch['image2'].to(DEVICE)
            elev1 = batch['elevation1'].to(DEVICE)
            targets = batch['elevation2'].to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(img1, img2, elev1)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            bs = targets.size(0)
            train_loss += loss.item() * bs
            train_mae += F.l1_loss(preds, targets, reduction='sum').item()
            n += bs
            
            pbar.set_postfix({
                'loss': f'{train_loss/n:.4f}',
                'mae_ft': f'{(train_mae/n)*train_dataset.elev_std:.3f}'
            })
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                img1 = batch['image1'].to(DEVICE)
                img2 = batch['image2'].to(DEVICE)
                elev1 = batch['elevation1'].to(DEVICE)
                
                preds = model(img1, img2, elev1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch['elevation2'].numpy())
        
        # Denormalize
        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        scheduler.step()
        
        train_mae_ft = (train_mae / n) * train_dataset.elev_std
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / n,
            'train_mae_ft': train_mae_ft,
            'val_mae_ft': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
        })
        
        print(f"   Val MAE: {val_metrics['mae']:.3f} ft | Val R²: {val_metrics['r2']:.3f}")
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save({
                'model_state_dict': model.state_dict(),
                'elev_mean': train_dataset.elev_mean,
                'elev_std': train_dataset.elev_std,
                'config': config,
                'epoch': epoch,
            }, output_dir / 'best_model.pt')
            print(f'   ✓ Saved best model!')
    
    # Test evaluation
    print('\n Evaluating on TEST set (unseen sites)...')
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            img1 = batch['image1'].to(DEVICE)
            img2 = batch['image2'].to(DEVICE)
            elev1 = batch['elevation1'].to(DEVICE)
            
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
    plt.savefig(output_dir / 'test_predictions.png', dpi=150)
    
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
    
    with open(output_dir / 'results.json', 'w') as f:
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
        'val_csv': 'data/val_quality.csv',
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
