#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from deepwater.models import create_model
from deepwater.utils import compute_all_metrics, print_metrics
import pandas as pd
import torchvision.transforms as T
from PIL import Image

print("Starting evaluation...")

output_dir = Path('outputs/triplet_20251202_140534')
print(f'Loading from {output_dir}')

checkpoint = torch.load(output_dir / 'best_model.pt', map_location='cpu', weights_only=False)

model = create_model(model_type='triplet', backbone='vit_tiny_patch16_224', pretrained=False, feature_dim=192, hidden_dim=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('mps')
model.eval()

elev_mean = float(checkpoint['elev_mean'])
elev_std = float(checkpoint['elev_std'])
print(f"Loaded model. elev_mean={elev_mean:.2f}, elev_std={elev_std:.2f}")

test_df = pd.read_csv('data/test_triplet.csv')
print(f"Test samples: {len(test_df)}")

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

all_preds, all_targets = [], []
n = len(test_df)

print("Running inference...")
with torch.no_grad():
    for idx in range(200):
        i, j, k = np.random.randint(n), np.random.randint(n), np.random.randint(n)
        if i == j or j == k or i == k: 
            continue
        
        r1, r2, q = test_df.iloc[i], test_df.iloc[j], test_df.iloc[k]
        
        img1 = transform(Image.open(r1['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        img2 = transform(Image.open(r2['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        img3 = transform(Image.open(q['image_paths']).convert('RGB')).unsqueeze(0).to('mps')
        
        e1 = torch.tensor([float((r1['gage_height_ft'] - elev_mean) / elev_std)], dtype=torch.float32).to('mps')
        e2 = torch.tensor([float((r2['gage_height_ft'] - elev_mean) / elev_std)], dtype=torch.float32).to('mps')
        target = float(q['gage_height_ft'])
        
        pred_norm = model(img1, img2, img3, e1, e2)
        pred = float(pred_norm.cpu().numpy()[0]) * elev_std + elev_mean
        
        all_preds.append(pred)
        all_targets.append(target)

print(f"Got {len(all_preds)} predictions")

metrics = compute_all_metrics(np.array(all_preds), np.array(all_targets))
print_metrics(metrics, title='TRIPLET Test Results')

print()
print('='*50)
print('MODEL COMPARISON')
print('='*50)
print(f'Siamese MAE:  0.073 ft')
print(f'Triplet MAE:  {metrics["mae"]:.3f} ft')
print()
print(f'Siamese R2:   0.994')
print(f'Triplet R2:   {metrics["r2"]:.3f}')
