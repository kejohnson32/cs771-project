#!/usr/bin/env python3
"""
Train Siamese model with RGB input (3 channels).
Uses 1 reference image + 1 query image.

Usage:
    python scripts/train_siamese_rgb.py --epochs 15
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


class SiameseWaterLevelModel(nn.Module):
    """Siamese model: 1 reference + 1 query image."""
    
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        backbone_dim = 192  # ViT-Tiny output dimension
        
        self.elev_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, ref, query, elev):
        feat_ref = self.backbone(ref)
        feat_query = self.backbone(query)
        elev_emb = self.elev_embed(elev.unsqueeze(-1))
        combined = torch.cat([feat_ref, feat_query, elev_emb], dim=1)
        return self.head(combined).squeeze(-1)


class SameSitePairDataset(torch.utils.data.Dataset):
    """Dataset that creates pairs from same camera site."""
    
    def __init__(self, csv_path, elev_mean=None, elev_std=None, max_pairs=5000):
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
        
        # Group by site
        self.site_groups = {
            site: self.df[self.df['camera_id'] == site].reset_index(drop=True)
            for site in self.df['camera_id'].unique()
            if len(self.df[self.df['camera_id'] == site]) >= 2
        }
        
        # Create pairs
        self.pairs = []
        sites = list(self.site_groups.keys())
        np.random.seed(42)
        while len(self.pairs) < max_pairs:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            if len(site_df) >= 2:
                i, j = np.random.choice(len(site_df), 2, replace=False)
                self.pairs.append((site, i, j))
        print(f"Created {len(self.pairs)} pairs from {len(self.site_groups)} sites")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        site, i, j = self.pairs[idx]
        site_df = self.site_groups[site]
        ref_row, query_row = site_df.iloc[i], site_df.iloc[j]
        
        ref_img = self.transform(Image.open(ref_row['image_path']).convert('RGB'))
        query_img = self.transform(Image.open(query_row['image_path']).convert('RGB'))
        
        ref_elev = (ref_row['gage_height_ft'] - self.elev_mean) / self.elev_std
        query_elev = (query_row['gage_height_ft'] - self.elev_mean) / self.elev_std
        
        return {
            'ref': ref_img,
            'query': query_img,
            'ref_elevation': torch.tensor(ref_elev, dtype=torch.float32),
            'query_elevation': torch.tensor(query_elev, dtype=torch.float32),
        }


def train(config):
    print('\n' + '='*60)
    print('SIAMESE RGB MODEL TRAINING')
    print('='*60)
    
    # Datasets
    train_ds = SameSitePairDataset(config['train_csv'], max_pairs=config['max_pairs'])
    val_ds = SameSitePairDataset(config['val_csv'], train_ds.elev_mean, train_ds.elev_std, max_pairs=1500)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])
    
    # Model
    model = SiameseWaterLevelModel().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    output_dir = Path('models/siamese_rgb')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            ref = batch['ref'].to(DEVICE)
            query = batch['query'].to(DEVICE)
            ref_elev = batch['ref_elevation'].to(DEVICE)
            target = batch['query_elevation'].to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(ref, query, ref_elev)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['ref'].to(DEVICE), batch['query'].to(DEVICE), 
                           batch['ref_elevation'].to(DEVICE))
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch['query_elevation'].numpy())
        
        val_preds = np.array(val_preds) * train_ds.elev_std + train_ds.elev_mean
        val_targets = np.array(val_targets) * train_ds.elev_std + train_ds.elev_mean
        val_mae = np.mean(np.abs(val_preds - val_targets))
        
        scheduler.step()
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_mae:.3f} ft")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'elev_mean': train_ds.elev_mean,
                'elev_std': train_ds.elev_std
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model!")
    
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
        'max_pairs': 5000
    })
