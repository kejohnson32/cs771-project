#!/usr/bin/env python3
"""
Simple Seen vs Unseen Evaluation

Run after create_seen_eval.py to evaluate models on:
- SEEN: New images from training sites (data/seen_eval.csv)
- UNSEEN: Completely new sites (data/unseen_eval.csv)

Usage:
    python scripts/eval_simple.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# SETUP
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEN_CSV = DATA_DIR / "seen_eval.csv"
UNSEEN_CSV = DATA_DIR / "unseen_eval.csv"

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# MODEL DEFINITIONS (must match training scripts exactly)
# =============================================================================

class SiameseWaterLevelModel(nn.Module):
    """Siamese model from train_siamese_rgb.py"""
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.elev_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, ref, query, elev):
        feat_ref = self.backbone(ref)
        feat_query = self.backbone(query)
        elev_emb = self.elev_embed(elev.unsqueeze(-1))
        combined = torch.cat([feat_ref, feat_query, elev_emb], dim=1)
        return self.head(combined).squeeze(-1)


class TripletWaterLevelModel(nn.Module):
    """Triplet model from train_triplet_rgb.py"""
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.cross_attn = nn.MultiheadAttention(192, num_heads=4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + 64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
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


class SiameseSAMModel(nn.Module):
    """Siamese SAM model (4 channels)"""
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        base = timm.create_model(backbone, pretrained=True, num_classes=0)
        old = base.patch_embed.proj
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight[:, :1]
            new.bias = old.bias
        base.patch_embed.proj = new
        self.backbone = base
        self.elev_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, ref, query, elev):
        feat_ref = self.backbone(ref)
        feat_query = self.backbone(query)
        elev_emb = self.elev_embed(elev.unsqueeze(-1))
        return self.head(torch.cat([feat_ref, feat_query, elev_emb], dim=1)).squeeze(-1)


class TripletSAMModel(nn.Module):
    """Triplet SAM model (4 channels)"""
    def __init__(self, backbone='vit_tiny_patch16_224'):
        super().__init__()
        base = timm.create_model(backbone, pretrained=True, num_classes=0)
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


# =============================================================================
# DATASETS
# =============================================================================

class PairDataset(Dataset):
    """Dataset for Siamese models"""
    def __init__(self, csv_path, elev_mean, elev_std, channels=3, max_pairs=3000):
        self.df = pd.read_csv(csv_path)
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & (self.df['gage_height_ft'] < 50)]
        
        self.elev_mean = elev_mean
        self.elev_std = elev_std
        self.channels = channels
        
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
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
        while len(self.pairs) < max_pairs and sites:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            if len(site_df) >= 2:
                i, j = np.random.choice(len(site_df), 2, replace=False)
                self.pairs.append((site, i, j))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        site, i, j = self.pairs[idx]
        site_df = self.site_groups[site]
        ref, query = site_df.iloc[i], site_df.iloc[j]
        
        ref_img = self.transform(Image.open(ref['image_path']).convert('RGB'))
        query_img = self.transform(Image.open(query['image_path']).convert('RGB'))
        
        if self.channels == 4:
            mask = torch.ones(1, 224, 224) * 0.5
            ref_img = torch.cat([ref_img, mask], dim=0)
            query_img = torch.cat([query_img, mask], dim=0)
        
        return {
            'ref': ref_img,
            'query': query_img,
            'ref_elev': torch.tensor((ref['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'query_elev': torch.tensor((query['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
        }


class TripletDataset(Dataset):
    """Dataset for Triplet models"""
    def __init__(self, csv_path, elev_mean, elev_std, channels=3, max_triplets=3000):
        self.df = pd.read_csv(csv_path)
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'], errors='coerce')
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        self.df = self.df[(self.df['gage_height_ft'] > 0) & (self.df['gage_height_ft'] < 50)]
        
        self.elev_mean = elev_mean
        self.elev_std = elev_std
        self.channels = channels
        
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.site_groups = {
            site: self.df[self.df['camera_id'] == site].reset_index(drop=True)
            for site in self.df['camera_id'].unique()
            if len(self.df[self.df['camera_id'] == site]) >= 3
        }
        
        self.triplets = []
        sites = list(self.site_groups.keys())
        np.random.seed(42)
        while len(self.triplets) < max_triplets and sites:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            if len(site_df) >= 3:
                i, j, k = np.random.choice(len(site_df), 3, replace=False)
                self.triplets.append((site, i, j, k))
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]
        r1, r2, q = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]
        
        img1 = self.transform(Image.open(r1['image_path']).convert('RGB'))
        img2 = self.transform(Image.open(r2['image_path']).convert('RGB'))
        img3 = self.transform(Image.open(q['image_path']).convert('RGB'))
        
        if self.channels == 4:
            mask = torch.ones(1, 224, 224) * 0.5
            img1 = torch.cat([img1, mask], dim=0)
            img2 = torch.cat([img2, mask], dim=0)
            img3 = torch.cat([img3, mask], dim=0)
        
        return {
            'ref1': img1, 'ref2': img2, 'query': img3,
            'elev1': torch.tensor((r1['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elev2': torch.tensor((r2['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
            'elev_query': torch.tensor((q['gage_height_ft'] - self.elev_mean) / self.elev_std, dtype=torch.float32),
        }


# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def evaluate_siamese(model, csv_path, elev_mean, elev_std, channels=3, desc="Eval"):
    ds = PairDataset(csv_path, elev_mean, elev_std, channels=channels)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pred = model(
                batch['ref'].to(DEVICE),
                batch['query'].to(DEVICE),
                batch['ref_elev'].to(DEVICE)
            )
            preds.extend((pred.cpu().numpy() * elev_std + elev_mean).tolist())
            targets.extend((batch['query_elev'].numpy() * elev_std + elev_mean).tolist())
    
    return compute_metrics(preds, targets), preds, targets


def evaluate_triplet(model, csv_path, elev_mean, elev_std, channels=3, desc="Eval"):
    ds = TripletDataset(csv_path, elev_mean, elev_std, channels=channels)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pred = model(
                batch['ref1'].to(DEVICE),
                batch['ref2'].to(DEVICE),
                batch['query'].to(DEVICE),
                batch['elev1'].to(DEVICE),
                batch['elev2'].to(DEVICE)
            )
            preds.extend((pred.cpu().numpy() * elev_std + elev_mean).tolist())
            targets.extend((batch['elev_query'].numpy() * elev_std + elev_mean).tolist())
    
    return compute_metrics(preds, targets), preds, targets


def plot_results(preds, targets, metrics, title, filepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, alpha=0.5, s=20)
    mn, mx = min(min(targets), min(preds)) - 0.5, max(max(targets), max(preds)) + 0.5
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('Actual (ft)', fontsize=12)
    ax.set_ylabel('Predicted (ft)', fontsize=12)
    ax.set_title(f"{title}\nMAE: {metrics['mae']:.3f} ft ({metrics['mae']*12:.1f} in), R²: {metrics['r2']:.3f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def load_model_with_remap(ckpt_path, model_class):
    """Load model, attempting to remap keys if needed."""
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = model_class().to(DEVICE)
    
    raw_state = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Try direct load first
    try:
        model.load_state_dict(raw_state, strict=True)
        return model, checkpoint['elev_mean'], checkpoint['elev_std'], "direct"
    except:
        pass
    
    # Try remapping keys
    remapped = {}
    for k, v in raw_state.items():
        new_k = k
        # backbone.model.* -> backbone.*
        if new_k.startswith("backbone.model."):
            new_k = "backbone." + new_k[len("backbone.model."):]
        # elevation_embed -> elev_embed
        elif new_k.startswith("elevation_embed."):
            idx_map = {'0': '0', '1': '2'}  # 0->0, 1->2 (ReLU is index 1)
            for old_idx, new_idx in idx_map.items():
                if f"elevation_embed.{old_idx}." in k:
                    new_k = k.replace(f"elevation_embed.{old_idx}.", f"elev_embed.{new_idx}.")
        # cross_attention -> cross_attn
        elif new_k.startswith("cross_attention."):
            new_k = "cross_attn." + new_k[len("cross_attention."):]
        # head remapping
        elif new_k.startswith("head.head."):
            idx_map = {'0': '0', '1': '0', '4': '3', '5': '3'}
            for old_idx, new_idx in idx_map.items():
                if f"head.head.{old_idx}." in k:
                    new_k = k.replace(f"head.head.{old_idx}.", f"head.{new_idx}.")
        elif new_k.startswith("head.output."):
            new_k = "head.5." + new_k[len("head.output."):]
        
        if new_k in model_state and model_state[new_k].shape == v.shape:
            remapped[new_k] = v
    
    # Load what we can
    model.load_state_dict(remapped, strict=False)
    return model, checkpoint['elev_mean'], checkpoint['elev_std'], f"remapped ({len(remapped)}/{len(model_state)} params)"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SEEN vs UNSEEN EVALUATION")
    print("="*70)
    
    # Check data exists
    if not SEEN_CSV.exists():
        print(f"\nERROR: {SEEN_CSV} not found!")
        print("Run: python scripts/create_seen_eval.py")
        return
    
    if not UNSEEN_CSV.exists():
        print(f"\nERROR: {UNSEEN_CSV} not found!")
        print("Run: python scripts/create_seen_eval.py")
        return
    
    seen_df = pd.read_csv(SEEN_CSV)
    unseen_df = pd.read_csv(UNSEEN_CSV)
    
    print(f"\nSEEN:   {len(seen_df)} images from {seen_df['camera_id'].nunique()} sites")
    print(f"UNSEEN: {len(unseen_df)} images from {unseen_df['camera_id'].nunique()} sites")
    
    results = {}
    
    # Define models to evaluate
    models_config = [
        ('siamese_rgb', SiameseWaterLevelModel, evaluate_siamese, 3),
        ('siamese_sam', SiameseSAMModel, evaluate_siamese, 4),
        ('triplet_rgb', TripletWaterLevelModel, evaluate_triplet, 3),
        ('triplet_sam', TripletSAMModel, evaluate_triplet, 4),
    ]
    
    for name, model_class, eval_fn, channels in models_config:
        ckpt_path = MODELS_DIR / name / 'best_model.pt'
        
        if not ckpt_path.exists():
            print(f"\n[{name}] Model not found at {ckpt_path}, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {name.upper()}")
        print(f"{'='*70}")
        
        model, elev_mean, elev_std, load_status = load_model_with_remap(ckpt_path, model_class)
        model.eval()
        print(f"Model loaded: {load_status}")
        print(f"Elevation normalization: mean={elev_mean:.2f}, std={elev_std:.2f}")
        
        results[name] = {}
        
        # Evaluate SEEN
        print(f"\n  --- SEEN (new images from training sites) ---")
        seen_metrics, seen_preds, seen_targets = eval_fn(
            model, str(SEEN_CSV), elev_mean, elev_std, channels, "SEEN"
        )
        results[name]['seen'] = seen_metrics
        print(f"  MAE: {seen_metrics['mae']:.3f} ft ({seen_metrics['mae']*12:.1f} in), R²: {seen_metrics['r2']:.3f}")
        
        plot_results(seen_preds, seen_targets, seen_metrics, 
                    f"{name} - SEEN", RESULTS_DIR / f"{name}_seen.png")
        
        # Evaluate UNSEEN
        print(f"\n  --- UNSEEN (completely new sites) ---")
        unseen_metrics, unseen_preds, unseen_targets = eval_fn(
            model, str(UNSEEN_CSV), elev_mean, elev_std, channels, "UNSEEN"
        )
        results[name]['unseen'] = unseen_metrics
        print(f"  MAE: {unseen_metrics['mae']:.3f} ft ({unseen_metrics['mae']*12:.1f} in), R²: {unseen_metrics['r2']:.3f}")
        
        plot_results(unseen_preds, unseen_targets, unseen_metrics,
                    f"{name} - UNSEEN", RESULTS_DIR / f"{name}_unseen.png")
        
        # Combined plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        ax1.scatter(seen_targets, seen_preds, alpha=0.5, c='green', label='Seen')
        mn, mx = 0, max(max(seen_targets), max(seen_preds)) + 1
        ax1.plot([mn, mx], [mn, mx], 'r--', lw=2)
        ax1.set_xlabel('Actual (ft)')
        ax1.set_ylabel('Predicted (ft)')
        ax1.set_title(f"SEEN: MAE={seen_metrics['mae']:.2f}ft, R²={seen_metrics['r2']:.3f}")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.scatter(unseen_targets, unseen_preds, alpha=0.5, c='blue', label='Unseen')
        ax2.plot([mn, mx], [mn, mx], 'r--', lw=2)
        ax2.set_xlabel('Actual (ft)')
        ax2.set_ylabel('Predicted (ft)')
        ax2.set_title(f"UNSEEN: MAE={unseen_metrics['mae']:.2f}ft, R²={unseen_metrics['r2']:.3f}")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"{name.upper()} - Seen vs Unseen", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{name}_comparison.png", dpi=150)
        plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Model':<15} {'Seen MAE':<12} {'Seen R²':<10} {'Unseen MAE':<12} {'Unseen R²':<10}")
    print("-"*60)
    for name, r in results.items():
        seen = r.get('seen', {})
        unseen = r.get('unseen', {})
        print(f"{name:<15} {seen.get('mae', 0):.3f} ft     {seen.get('r2', 0):.3f}      {unseen.get('mae', 0):.3f} ft      {unseen.get('r2', 0):.3f}")
    
    # Save results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_DIR}/results.json")
    
    # Create comparison bar chart
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = list(results.keys())
        seen_maes = [results[m].get('seen', {}).get('mae', 0) for m in models]
        unseen_maes = [results[m].get('unseen', {}).get('mae', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1 = axes[0]
        ax1.bar(x - width/2, seen_maes, width, label='Seen', color='green', alpha=0.7)
        ax1.bar(x + width/2, unseen_maes, width, label='Unseen', color='blue', alpha=0.7)
        ax1.set_ylabel('MAE (feet)')
        ax1.set_title('MAE Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        seen_r2s = [results[m].get('seen', {}).get('r2', 0) for m in models]
        unseen_r2s = [results[m].get('unseen', {}).get('r2', 0) for m in models]
        
        ax2 = axes[1]
        ax2.bar(x - width/2, seen_r2s, width, label='Seen', color='green', alpha=0.7)
        ax2.bar(x + width/2, unseen_r2s, width, label='Unseen', color='blue', alpha=0.7)
        ax2.set_ylabel('R²')
        ax2.set_title('R² Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=150)
        plt.close()
        print(f"Comparison chart saved to {RESULTS_DIR}/model_comparison.png")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    main()