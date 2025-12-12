#!/usr/bin/env python3
"""
Train Triplet model with RGB input (3 channels).
Uses 2 reference images + 1 query image with cross-attention.

Usage:
    python scripts/train_triplet_rgb.py --epochs 15
"""

import os
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
import json
import matplotlib.pyplot as plt

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


class TripletWaterLevelModel(nn.Module):
    """Triplet model: 2 references + 1 query with cross-attention."""
    
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        backbone_dim = 192  # ViT-Tiny
        
        self.cross_attn = nn.MultiheadAttention(backbone_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, ref1, ref2, query, elev1, elev2):
        feat1 = self.backbone(ref1)
        feat2 = self.backbone(ref2)
        feat_q = self.backbone(query)
        
        # Cross-attention: query attends to references
        refs = torch.stack([feat1, feat2], dim=1)  # [B, 2, D]
        attn_out, _ = self.cross_attn(feat_q.unsqueeze(1), refs, refs)
        
        # Elevation embedding
        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        
        # Combine and predict
        combined = torch.cat([feat_q, attn_out.squeeze(1), elev_emb], dim=1)
        return self.head(combined).squeeze(-1)


class SameSiteTripletDataset(torch.utils.data.Dataset):
    """Dataset that creates triplets from same camera site."""
    
    def __init__(self, csv_path, elev_mean=None, elev_std=None, min_elev_diff=0.3, max_triplets=5000):
        self.df = pd.read_csv(csv_path)
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & (self.df['gage_height_ft'] < 50)]
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.elev_mean = elev_mean if elev_mean else self.df['gage_height_ft'].mean()
        self.elev_std = elev_std if elev_std else self.df['gage_height_ft'].std()
        print(f"Elevation normalization: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")
        
        # Group by site
        self.site_groups = {
            site: self.df[self.df['camera_id'] == site].reset_index(drop=True)
            for site in self.df['camera_id'].unique()
            if len(self.df[self.df['camera_id'] == site]) >= 3
        }
        print(f"Sites with 3+ images: {len(self.site_groups)}")
        
        # Create triplets with elevation difference constraint
        self.triplets = []
        sites = list(self.site_groups.keys())
        np.random.seed(42)
        attempts = 0
        while len(self.triplets) < max_triplets and attempts < max_triplets * 20:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            n = len(site_df)
            if n >= 3:
                i, j, k = np.random.choice(n, 3, replace=False)
                elev_diff = abs(site_df.iloc[i]['gage_height_ft'] - site_df.iloc[j]['gage_height_ft'])
                if elev_diff >= min_elev_diff:
                    self.triplets.append((site, i, j, k))
            attempts += 1
        print(f"Created {len(self.triplets)} triplets")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]
        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]
        
        img1 = self.transform(Image.open(row1['image_path']).convert('RGB'))
        img2 = self.transform(Image.open(row2['image_path']).convert('RGB'))
        img3 = self.transform(Image.open(row3['image_path']).convert('RGB'))
        
        return {
            'ref1': img1,
            'ref2': img2,
            'query': img3,
            'elevation1': torch.tensor((row1['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elevation2': torch.tensor((row2['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elevation_query': torch.tensor((row3['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
        }


def train(config):
    print('\n' + '='*60)
    print('TRIPLET RGB MODEL TRAINING')
    print('='*60)
    
    # Datasets
    train_ds = SameSiteTripletDataset(config['train_csv'], max_triplets=config['max_triplets'])
    val_ds = SameSiteTripletDataset(config['val_csv'], train_ds.elev_mean, train_ds.elev_std, max_triplets=1500)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])
    
    # Model
    model = TripletWaterLevelModel().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    output_dir = Path('models/triplet_rgb')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    train_losses, val_maes = [], []
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            ref1 = batch['ref1'].to(DEVICE)
            ref2 = batch['ref2'].to(DEVICE)
            query = batch['query'].to(DEVICE)
            elev1 = batch['elevation1'].to(DEVICE)
            elev2 = batch['elevation2'].to(DEVICE)
            target = batch['elevation_query'].to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(ref1, ref2, query, elev1, elev2)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_losses.append(train_loss / n_batches)
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE),
                    batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE)
                )
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch['elevation_query'].numpy())
        
        # Convert back to feet
        val_preds = np.array(val_preds) * train_ds.elev_std + train_ds.elev_mean
        val_targets = np.array(val_targets) * train_ds.elev_std + train_ds.elev_mean
        val_mae = np.mean(np.abs(val_preds - val_targets))
        val_maes.append(val_mae)
        
        scheduler.step()
        print(f"  Train Loss: {train_losses[-1]:.4f} | Val MAE: {val_mae:.3f} ft")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'elev_mean': train_ds.elev_mean,
                'elev_std': train_ds.elev_std
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model!")
    
    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[1].plot(val_maes)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation MAE (ft)')
    axes[1].set_title('Validation MAE')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    
    print(f"\nTraining complete! Best Val MAE: {best_val_mae:.3f} ft")
    print(f"Model saved to: {output_dir}/best_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train({
        'train_csv': 'data/train_quality.csv',
        'val_csv': 'data/val_quality.csv',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_triplets': 5000
    })
