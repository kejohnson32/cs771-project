#!/usr/bin/env python3
"""
Train Triplet model with SAM mask (4 channels: RGB + mask).

Usage:
    python scripts/train_triplet_sam.py --epochs 15
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm
from PIL import Image
import torchvision.transforms as T

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# SAM2 setup
def get_sam2_predictor():
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        checkpoint = os.environ.get('SAM2_CHECKPOINT', 'sam2/checkpoints/sam2.1_hiera_small.pt')
        config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        model = build_sam2(config, checkpoint, device=DEVICE)
        return SAM2ImagePredictor(model)
    except:
        print("SAM2 not available, using dummy masks")
        return None


class TripletSAMModel(nn.Module):
    """Triplet model with 4-channel input (RGB + SAM mask)."""
    
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        base = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        # Modify to accept 4 channels
        old = base.patch_embed.proj
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight[:, :1]
            new.bias = old.bias
        base.patch_embed.proj = new
        self.backbone = base
        
        self.cross_attn = nn.MultiheadAttention(192, 4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + 64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, ref1, ref2, query, elev1, elev2):
        f1, f2, fq = self.backbone(ref1), self.backbone(ref2), self.backbone(query)
        refs = torch.stack([f1, f2], dim=1)
        attn_out, _ = self.cross_attn(fq.unsqueeze(1), refs, refs)
        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        return self.head(torch.cat([fq, attn_out.squeeze(1), elev_emb], dim=1)).squeeze(-1)


class SameSiteTripletDatasetSAM(torch.utils.data.Dataset):
    """Dataset with SAM masks."""
    
    def __init__(self, csv_path, elev_mean=None, elev_std=None, max_triplets=5000,
                 mask_cache_dir='data/mask_cache'):
        self.df = pd.read_csv(csv_path)
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & (self.df['gage_height_ft'] < 50)]
        
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
        
        self.elev_mean = elev_mean if elev_mean else self.df['gage_height_ft'].mean()
        self.elev_std = elev_std if elev_std else self.df['gage_height_ft'].std()
        
        self.mask_cache_dir = Path(mask_cache_dir)
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sam_predictor = get_sam2_predictor()
        
        self.site_groups = {
            site: self.df[self.df['camera_id'] == site].reset_index(drop=True)
            for site in self.df['camera_id'].unique()
            if len(self.df[self.df['camera_id'] == site]) >= 3
        }
        
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
                if abs(site_df.iloc[i]['gage_height_ft'] - site_df.iloc[j]['gage_height_ft']) >= 0.3:
                    self.triplets.append((site, i, j, k))
            attempts += 1
        print(f"Created {len(self.triplets)} triplets")
    
    def _generate_mask(self, image_np):
        if self.sam_predictor is None:
            return np.ones((224, 224), dtype=np.float32)
        try:
            self.sam_predictor.set_image(image_np)
            h, w = image_np.shape[:2]
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array([[w//2, int(h*0.7)]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            mask = masks[np.argmax(scores)]
            return np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((224,224))) / 255.0
        except:
            return np.ones((224, 224), dtype=np.float32)
    
    def _load_image_with_mask(self, path):
        img = Image.open(path).convert('RGB')
        img_resized = self.resize(img)
        img_np = np.array(img_resized)
        
        cache_path = self.mask_cache_dir / (Path(path).stem + '_mask.npy')
        if cache_path.exists():
            mask = np.load(cache_path)
        else:
            mask = self._generate_mask(img_np)
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
            'elevation1': torch.tensor((row1['gage_height_ft']-self.elev_mean)/self.elev_std, dtype=torch.float32),
            'elevation2': torch.tensor((row2['gage_height_ft']-self.elev_mean)/self.elev_std, dtype=torch.float32),
            'elevation_query': torch.tensor((row3['gage_height_ft']-self.elev_mean)/self.elev_std, dtype=torch.float32),
        }


def train(config):
    print('\n' + '='*60)
    print('TRIPLET SAM MODEL TRAINING')
    print('='*60)
    
    train_ds = SameSiteTripletDatasetSAM(config['train_csv'], max_triplets=config['max_triplets'])
    val_ds = SameSiteTripletDatasetSAM(config['val_csv'], train_ds.elev_mean, train_ds.elev_std, max_triplets=1500)
    
    batch_size = max(1, config['batch_size'] // 2)  # Smaller batch for 4-channel
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    
    model = TripletSAMModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    output_dir = Path('models/triplet_sam')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            pred = model(
                batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE),
                batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE)
            )
            loss = F.mse_loss(pred, batch['elevation_query'].to(DEVICE))
            loss.backward()
            optimizer.step()
        
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
        
        val_preds = np.array(val_preds) * train_ds.elev_std + train_ds.elev_mean
        val_targets = np.array(val_targets) * train_ds.elev_std + train_ds.elev_mean
        val_mae = np.mean(np.abs(val_preds - val_targets))
        
        scheduler.step()
        print(f"  Val MAE: {val_mae:.3f} ft")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'elev_mean': train_ds.elev_mean,
                'elev_std': train_ds.elev_std
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved!")
    
    print(f"\nBest Val MAE: {best_val_mae:.3f} ft")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
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
