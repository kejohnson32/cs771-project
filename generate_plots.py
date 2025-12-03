#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from deepwater.models import create_model
from deepwater.utils import compute_all_metrics
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

print("Generating Triplet plot...")

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load triplet model
triplet_dir = Path('outputs/triplet_20251202_140534')
triplet_ckpt = torch.load(triplet_dir / 'best_model.pt', map_location='cpu', weights_only=False)
triplet_model = create_model(model_type='triplet', backbone='vit_tiny_patch16_224', pretrained=False, feature_dim=192, hidden_dim=128)
triplet_model.load_state_dict(triplet_ckpt['model_state_dict'])
triplet_model.to('mps')
triplet_model.eval()

elev_mean = float(triplet_ckpt['elev_mean'])
elev_std = float(triplet_ckpt['elev_std'])

test_df = pd.read_csv('data/test_triplet.csv')
n = len(test_df)

triplet_preds, triplet_targets = [], []

print("Running inference...")
with torch.no_grad():
    for _ in range(300):
        i, j, k = np.random.randint(n), np.random.randint(n), np.random.randint(n)
        if i == j or j == k or i == k: continue
        
        r1, r2, q = test_df.iloc[i], test_df.iloc[j], test_df.iloc[k]
        
        img1 = transform(Image.open(r1['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        img2 = transform(Image.open(r2['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        img3 = transform(Image.open(q['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        
        e1 = torch.tensor([float((r1['gage_height_ft'] - elev_mean) / elev_std)], dtype=torch.float32).to('mps')
        e2 = torch.tensor([float((r2['gage_height_ft'] - elev_mean) / elev_std)], dtype=torch.float32).to('mps')
        target = float(q['gage_height_ft'])
        
        pred_norm = triplet_model(img1, img2, img3, e1, e2)
        pred = float(pred_norm.cpu().numpy()[0]) * elev_std + elev_mean
        
        triplet_preds.append(pred)
        triplet_targets.append(target)

triplet_preds = np.array(triplet_preds)
triplet_targets = np.array(triplet_targets)
metrics = compute_all_metrics(triplet_preds, triplet_targets)

print(f"MAE: {metrics['mae']:.3f}, R2: {metrics['r2']:.3f}")

# Create plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(triplet_targets, triplet_preds, alpha=0.5, s=30, c='#4CAF50', label='Predictions')
mn, mx = min(triplet_targets.min(), triplet_preds.min()) - 0.2, max(triplet_targets.max(), triplet_preds.max()) + 0.2
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Water Level (ft)', fontsize=12)
ax.set_ylabel('Predicted Water Level (ft)', fontsize=12)
ax.set_title(f'Triplet Model (3-Image)\nMAE: {metrics["mae"]:.3f} ft, R²: {metrics["r2"]:.3f}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
plt.tight_layout()
plt.savefig('outputs/triplet_predictions.png', dpi=150)
print("Saved outputs/triplet_predictions.png")
