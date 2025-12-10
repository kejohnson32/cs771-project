#!/usr/bin/env python3
"""
Train Siamese model with SAM2 water masks (RGB + Mask = 4 channels).

This model uses SAM2 to segment water regions and adds the mask as a 4th channel,
helping the model focus on water level changes rather than background.

Usage:
    python train_siamese_sam.py --epochs 30

Requirements:
    - SAM2 checkpoint at: sam2/checkpoints/sam2.1_hiera_small.pt
    - Or set SAM2_CHECKPOINT environment variable
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

SAM2_CHECKPOINT = os.environ.get(
    'SAM2_CHECKPOINT', 
    'sam2/checkpoints/sam2.1_hiera_small.pt'
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_sam2_predictor = None

def get_sam2_predictor():
    """Lazy load SAM2 predictor."""
    global _sam2_predictor
    if _sam2_predictor is None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            print(f"Loading SAM2 from {SAM2_CHECKPOINT}...")
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
            _sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load SAM2: {e}")
            print("Will use dummy masks (all ones)")
            _sam2_predictor = "dummy"
    return _sam2_predictor


def generate_water_mask(image_np, predictor):
    """
    Generate water segmentation mask using SAM2.
    
    Args:
        image_np: RGB image as numpy array (H, W, 3)
        predictor: SAM2ImagePredictor or "dummy"
    
    Returns:
        Binary mask (H, W) with 1 for water, 0 for non-water
    """
    if predictor == "dummy":
        # Return mask of all ones (water everywhere)
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    
    try:
        predictor.set_image(image_np)
        
        # Use center point as prompt (assuming water is in center-bottom)
        h, w = image_np.shape[:2]
        point_coords = np.array([[w // 2, int(h * 0.7)]])  # Center-bottom
        point_labels = np.array([1])  # Foreground
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Take best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.float32)
        
        return mask
        
    except Exception as e:
        # Fallback to all ones
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)


# =============================================================================
# MODEL (4-channel input: RGB + Mask)
# =============================================================================

class SiameseSAMModel(nn.Module):
    """Siamese network with 4-channel input (RGB + water mask)."""
    
    def __init__(self, backbone='vit_tiny_patch16_224', feature_dim=192,
                 hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Load pretrained backbone
        base_model = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        # Get original conv layer
        if hasattr(base_model, 'patch_embed'):
            # ViT models
            old_proj = base_model.patch_embed.proj
            # Create new 4-channel projection
            new_proj = nn.Conv2d(
                4, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding
            )
            # Initialize: copy RGB weights, init mask channel
            with torch.no_grad():
                new_proj.weight[:, :3, :, :] = old_proj.weight
                new_proj.weight[:, 3:, :, :] = old_proj.weight[:, :1, :, :]  # Copy R channel
                new_proj.bias = old_proj.bias
            base_model.patch_embed.proj = new_proj
        else:
            # CNN models
            old_conv = base_model.conv1 if hasattr(base_model, 'conv1') else base_model.stem[0]
            new_conv = nn.Conv2d(
                4, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight[:, :1, :, :]
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            if hasattr(base_model, 'conv1'):
                base_model.conv1 = new_conv
            else:
                base_model.stem[0] = new_conv
        
        self.backbone = base_model
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 4, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]
        
        # Elevation embedding
        self.elev_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Prediction head
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
            img1, img2: 4-channel images [B, 4, 224, 224]
            elev1: Reference elevation [B]
        """
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        elev_emb = self.elev_embed(elev1.unsqueeze(-1))
        combined = torch.cat([feat1, feat2, elev_emb], dim=1)
        return self.head(combined).squeeze(-1)


# =============================================================================
# DATASET WITH SAM MASKS
# =============================================================================

class SameSitePairDatasetSAM(torch.utils.data.Dataset):
    """Dataset with SAM water masks as 4th channel."""
    
    def __init__(self, csv_path, elev_mean=None, elev_std=None,
                 min_elev_diff=0.2, max_pairs_per_site=2000,
                 cache_masks=True, mask_cache_dir='data/mask_cache'):
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Clean data
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & 
                          (self.df['gage_height_ft'] < 50)].reset_index(drop=True)
        
        # RGB transforms (no normalization yet - need to add mask first)
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],  # 4 channels
            std=[0.229, 0.224, 0.225, 0.5]
        )
        
        if elev_mean is None:
            self.elev_mean = self.df['gage_height_ft'].mean()
            self.elev_std = self.df['gage_height_ft'].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std
        
        print(f"Elevation normalization: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")
        
        # Mask caching
        self.cache_masks = cache_masks
        self.mask_cache_dir = Path(mask_cache_dir)
        if cache_masks:
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get SAM predictor
        self.sam_predictor = get_sam2_predictor()
        
        # Group by site
        self.site_groups = {}
        for site in self.df['camera_id'].unique():
            site_df = self.df[self.df['camera_id'] == site].reset_index(drop=True)
            if len(site_df) >= 2:
                self.site_groups[site] = site_df
        
        print(f"Sites with 2+ images: {len(self.site_groups)}")
        
        # Build pairs
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
            
            if len(site_pairs) > max_pairs_per_site:
                np.random.shuffle(site_pairs)
                site_pairs = site_pairs[:max_pairs_per_site]
            
            self.pairs.extend(site_pairs)
            print(f"  {site}: {len(site_pairs)} pairs")
        
        print(f"Total pairs: {len(self.pairs)}")
    
    def _get_mask_cache_path(self, image_path):
        """Get cache path for mask."""
        # Create unique cache filename based on image path
        cache_name = Path(image_path).stem + '_mask.npy'
        site = Path(image_path).parent.parent.name
        cache_dir = self.mask_cache_dir / site
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_name
    
    def _load_image_with_mask(self, image_path):
        """Load image and generate/load cached mask."""
        # Load RGB image
        img = Image.open(image_path).convert('RGB')
        img_resized = self.resize(img)
        img_np = np.array(img_resized)
        
        # Check mask cache
        if self.cache_masks:
            cache_path = self._get_mask_cache_path(image_path)
            if cache_path.exists():
                mask = np.load(cache_path)
            else:
                # Generate mask
                mask = generate_water_mask(img_np, self.sam_predictor)
                # Resize mask to 224x224
                mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                np.save(cache_path, mask.astype(np.float32))
        else:
            mask = generate_water_mask(img_np, self.sam_predictor)
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))) / 255.0
        
        # Convert to tensor
        img_tensor = self.to_tensor(img_resized)  # [3, 224, 224]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # [1, 224, 224]
        
        # Concatenate to 4 channels
        img_4ch = torch.cat([img_tensor, mask_tensor], dim=0)  # [4, 224, 224]
        
        # Normalize
        img_4ch = self.normalize(img_4ch)
        
        return img_4ch
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        site, i, j = self.pairs[idx]
        site_df = self.site_groups[site]
        
        row1 = site_df.iloc[i]
        row2 = site_df.iloc[j]
        
        img1 = self._load_image_with_mask(row1['image_path'])
        img2 = self._load_image_with_mask(row2['image_path'])
        
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
    preds = np.array(preds)
    targets = np.array(targets)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': corr}


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    print('\n' + '='*70)
    print(' SIAMESE + SAM MODEL TRAINING (RGB + Mask)')
    print('='*70)
    
    # Create datasets
    print('\n Loading datasets with SAM masks...')
    train_dataset = SameSitePairDatasetSAM(
        config['train_csv'],
        min_elev_diff=config['min_elev_diff'],
        max_pairs_per_site=config['max_pairs_per_site'],
        cache_masks=True
    )
    
    val_dataset = SameSitePairDatasetSAM(
        config['val_csv'],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        min_elev_diff=config['min_elev_diff'],
        max_pairs_per_site=config['max_pairs_per_site'],
        cache_masks=True
    )
    
    test_dataset = SameSitePairDatasetSAM(
        config['test_csv'],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        min_elev_diff=config['min_elev_diff'],
        max_pairs_per_site=config['max_pairs_per_site'],
        cache_masks=True
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
    print('\n🔧 Creating 4-channel model...')
    model = SiameseSAMModel(
        backbone=config['backbone'],
        feature_dim=192,
        hidden_dim=128,
        dropout=0.1
    )
    model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'   Parameters: {n_params/1e6:.2f}M')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f'siamese_sam_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print('\n Training...')
    best_val_mae = float('inf')
    
    for epoch in range(config['num_epochs']):
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
        
        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        scheduler.step()
        
        print(f"   Val MAE: {val_metrics['mae']:.3f} ft | Val R²: {val_metrics['r2']:.3f}")
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save({
                'model_state_dict': model.state_dict(),
                'elev_mean': train_dataset.elev_mean,
                'elev_std': train_dataset.elev_std,
                'config': config,
            }, output_dir / 'best_model.pt')
            print(f'   ✓ Saved best model!')
    
    # Test
    print('\n Evaluating on TEST set...')
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
    ax.scatter(test_targets_ft, test_preds_ft, alpha=0.5, s=20, c='#9C27B0')
    
    mn = min(test_targets_ft.min(), test_preds_ft.min()) - 0.5
    mx = max(test_targets_ft.max(), test_preds_ft.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    
    ax.set_xlabel('Actual Water Level (ft)', fontsize=12)
    ax.set_ylabel('Predicted Water Level (ft)', fontsize=12)
    ax.set_title(f'Siamese + SAM Model\n'
                 f'MAE: {test_metrics["mae"]:.3f} ft ({test_metrics["mae"]*12:.1f} in), '
                 f'R²: {test_metrics["r2"]:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_predictions.png', dpi=150)
    
    # Save results
    import json
    results = {
        'model': 'siamese_sam',
        'test_mae_ft': test_metrics['mae'],
        'test_mae_inches': test_metrics['mae'] * 12,
        'test_rmse_ft': test_metrics['rmse'],
        'test_r2': test_metrics['r2'],
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n' + '='*70)
    print(' SIAMESE + SAM RESULTS')
    print('='*70)
    print(f'Test MAE:  {test_metrics["mae"]:.3f} ft ({test_metrics["mae"]*12:.1f} inches)')
    print(f'Test RMSE: {test_metrics["rmse"]:.3f} ft')
    print(f'Test R²:   {test_metrics["r2"]:.3f}')
    print(f'Output:    {output_dir}')
    
    return test_metrics


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
