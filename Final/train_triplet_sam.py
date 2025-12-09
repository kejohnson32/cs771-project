#!/usr/bin/env python3
"""
Train Triplet model with SAM2 water masks (RGB + Mask = 4 channels).

Usage:
    python train_triplet_sam.py --epochs 30
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
# SAM2 SETUP
# =============================================================================

SAM2_CHECKPOINT = os.environ.get('SAM2_CHECKPOINT', 'sam2/checkpoints/sam2.1_hiera_small.pt')
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_sam2_predictor = None

def get_sam2_predictor():
    global _sam2_predictor
    if _sam2_predictor is None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print(f"Loading SAM2 from {SAM2_CHECKPOINT}...")
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
            _sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 loaded!")
        except Exception as e:
            print(f"Warning: Could not load SAM2: {e}")
            _sam2_predictor = "dummy"
    return _sam2_predictor


def generate_water_mask(image_np, predictor):
    if predictor == "dummy":
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    try:
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, int(h * 0.7)]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        return masks[np.argmax(scores)].astype(np.float32)
    except:
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)


# =============================================================================
# MODEL
# =============================================================================

class TripletSAMModel(nn.Module):
    def __init__(self, backbone='vit_tiny_patch16_224', hidden_dim=128, dropout=0.1):
        super().__init__()
        
        base_model = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        if hasattr(base_model, 'patch_embed'):
            old_proj = base_model.patch_embed.proj
            new_proj = nn.Conv2d(4, old_proj.out_channels, kernel_size=old_proj.kernel_size,
                                 stride=old_proj.stride, padding=old_proj.padding)
            with torch.no_grad():
                new_proj.weight[:, :3] = old_proj.weight
                new_proj.weight[:, 3:] = old_proj.weight[:, :1]
                new_proj.bias = old_proj.bias
            base_model.patch_embed.proj = new_proj
        
        self.backbone = base_model
        
        with torch.no_grad():
            backbone_dim = self.backbone(torch.randn(1, 4, 224, 224)).shape[1]
        
        self.cross_attn = nn.MultiheadAttention(backbone_dim, 4, dropout, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 64, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, ref1, ref2, query, elev1, elev2):
        feat1, feat2, feat_q = self.backbone(ref1), self.backbone(ref2), self.backbone(query)
        refs = torch.stack([feat1, feat2], dim=1)
        attn_out, _ = self.cross_attn(feat_q.unsqueeze(1), refs, refs)
        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        return self.head(torch.cat([feat_q, attn_out.squeeze(1), elev_emb], dim=1)).squeeze(-1)


# =============================================================================
# DATASET
# =============================================================================

class SameSiteTripletDatasetSAM(torch.utils.data.Dataset):
    def __init__(self, csv_path, elev_mean=None, elev_std=None, min_elev_diff=0.3, 
                 max_triplets=5000, cache_masks=True, mask_cache_dir='data/mask_cache'):
        
        self.df = pd.read_csv(csv_path)
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & (self.df['gage_height_ft'] < 50)].reset_index(drop=True)
        print(f"Loaded {len(self.df)} samples")
        
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        
        self.elev_mean = elev_mean if elev_mean else self.df['gage_height_ft'].mean()
        self.elev_std = elev_std if elev_std else self.df['gage_height_ft'].std()
        print(f"Normalization: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")
        
        self.cache_masks = cache_masks
        self.mask_cache_dir = Path(mask_cache_dir)
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sam_predictor = get_sam2_predictor()
        
        self.site_groups = {site: self.df[self.df['camera_id'] == site].reset_index(drop=True) 
                           for site in self.df['camera_id'].unique() 
                           if len(self.df[self.df['camera_id'] == site]) >= 3}
        print(f"Sites with 3+ images: {len(self.site_groups)}")
        
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
                if abs(site_df.iloc[i]['gage_height_ft'] - site_df.iloc[j]['gage_height_ft']) >= min_elev_diff:
                    self.triplets.append((site, i, j, k))
            attempts += 1
        print(f"Created {len(self.triplets)} triplets")
    
    def _load_image_with_mask(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_resized = self.resize(img)
        img_np = np.array(img_resized)
        
        cache_path = self.mask_cache_dir / Path(image_path).parent.parent.name / (Path(image_path).stem + '_mask.npy')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.cache_masks and cache_path.exists():
            mask = np.load(cache_path)
        else:
            mask = generate_water_mask(img_np, self.sam_predictor)
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))) / 255.0
            if self.cache_masks:
                np.save(cache_path, mask.astype(np.float32))
        
        img_tensor = self.to_tensor(img_resized)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return self.normalize(torch.cat([img_tensor, mask_tensor], dim=0))
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]
        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]
        
        return {
            'ref1': self._load_image_with_mask(row1['image_path']),
            'ref2': self._load_image_with_mask(row2['image_path']),
            'query': self._load_image_with_mask(row3['image_path']),
            'elevation1': torch.tensor((row1['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elevation2': torch.tensor((row2['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elevation_query': torch.tensor((row3['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
        }


def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res, ss_tot = np.sum((targets - preds) ** 2), np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0}


def train(config):
    print('\n' + '='*70)
    print('🌊 TRIPLET + SAM MODEL TRAINING')
    print('='*70)
    
    train_dataset = SameSiteTripletDatasetSAM(config['train_csv'], max_triplets=config['max_triplets'])
    val_dataset = SameSiteTripletDatasetSAM(config['val_csv'], train_dataset.elev_mean, train_dataset.elev_std, max_triplets=1500)
    test_dataset = SameSiteTripletDatasetSAM(config['test_csv'], train_dataset.elev_mean, train_dataset.elev_std, max_triplets=1500)
    
    batch_size = max(1, config['batch_size'] // 2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    model = TripletSAMModel(config['backbone']).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
    
    output_dir = Path('outputs') / f'triplet_sam_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss, n = 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        for batch in pbar:
            ref1, ref2, query = batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE)
            elev1, elev2, targets = batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE), batch['elevation_query'].to(DEVICE)
            
            optimizer.zero_grad()
            loss = F.mse_loss(model(ref1, ref2, query, elev1, elev2), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            n += targets.size(0)
            pbar.set_postfix({'loss': f'{train_loss/n:.4f}'})
            if DEVICE.type == 'mps': torch.mps.empty_cache()
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE),
                             batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE))
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch['elevation_query'].numpy())
        
        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        scheduler.step()
        
        print(f"   Val MAE: {val_metrics['mae']:.3f} ft | R²: {val_metrics['r2']:.3f}")
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save({'model_state_dict': model.state_dict(), 'elev_mean': train_dataset.elev_mean, 
                       'elev_std': train_dataset.elev_std}, output_dir / 'best_model.pt')
            print('   ✓ Saved!')
    
    # Test
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE),
                         batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE))
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(batch['elevation_query'].numpy())
    
    test_preds_ft = np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    test_targets_ft = np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(test_targets_ft, test_preds_ft, alpha=0.5, s=20, c='#FF5722')
    mn, mx = min(test_targets_ft.min(), test_preds_ft.min()) - 0.5, max(test_targets_ft.max(), test_preds_ft.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('Actual (ft)'); ax.set_ylabel('Predicted (ft)')
    ax.set_title(f'Triplet+SAM: MAE={test_metrics["mae"]:.3f}ft, R²={test_metrics["r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'test_predictions.png', dpi=150)
    
    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'model': 'triplet_sam', 'test_mae_ft': test_metrics['mae'], 
                  'test_mae_inches': test_metrics['mae']*12, 'test_r2': test_metrics['r2']}, f, indent=2)
    
    print('\n' + '='*70)
    print(f' TRIPLET+SAM: MAE={test_metrics["mae"]:.3f}ft ({test_metrics["mae"]*12:.1f}in), R²={test_metrics["r2"]:.3f}')
    print(f'Output: {output_dir}')
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train({
        'train_csv': 'data/train_quality.csv', 'val_csv': 'data/val_quality.csv', 'test_csv': 'data/test_quality.csv',
        'backbone': 'vit_tiny_patch16_224', 'batch_size': args.batch_size, 'num_epochs': args.epochs,
        'learning_rate': args.lr, 'max_triplets': 5000
    })
