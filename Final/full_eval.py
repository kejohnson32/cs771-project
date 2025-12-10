import torch
import torch.nn as nn
import numpy as np
import json
import timm
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = sorted(Path('outputs').glob('triplet_rgb_*'))[-1]
checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
elev_mean, elev_std = checkpoint['elev_mean'], checkpoint['elev_std']
print(f'Model: {output_dir}')

class TripletWaterLevelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        backbone_dim = 192
        self.cross_attn = nn.MultiheadAttention(backbone_dim, 4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(backbone_dim * 2 + 64, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, ref1, ref2, query, elev1, elev2):
        feat1, feat2, feat_q = self.backbone(ref1), self.backbone(ref2), self.backbone(query)
        refs = torch.stack([feat1, feat2], dim=1)
        attn_out, _ = self.cross_attn(feat_q.unsqueeze(1), refs, refs)
        elev_emb = self.elev_embed(torch.stack([elev1, elev2], dim=1))
        return self.head(torch.cat([feat_q, attn_out.squeeze(1), elev_emb], dim=1)).squeeze(-1)

model = TripletWaterLevelModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# Test evaluation
df = pd.read_csv('data/test_quality.csv')
site_groups = {s: g.reset_index(drop=True) for s, g in df.groupby('camera_id') if len(g) >= 3}
print(f'Test sites: {list(site_groups.keys())}')

preds, targets, sites_list = [], [], []
samples = []

for site, sdf in tqdm(site_groups.items(), desc='Testing'):
    n = len(sdf)
    for idx in range(min(50, n*(n-1)*(n-2)//6)):
        i, j, k = np.random.choice(n, 3, replace=False)
        r1, r2, q = sdf.iloc[i], sdf.iloc[j], sdf.iloc[k]
        img1 = transform(Image.open(r1['image_path']).convert('RGB')).unsqueeze(0).to(DEVICE)
        img2 = transform(Image.open(r2['image_path']).convert('RGB')).unsqueeze(0).to(DEVICE)
        img3 = transform(Image.open(q['image_path']).convert('RGB')).unsqueeze(0).to(DEVICE)
        e1 = torch.tensor([(r1['gage_height_ft']-elev_mean)/elev_std], dtype=torch.float32).to(DEVICE)
        e2 = torch.tensor([(r2['gage_height_ft']-elev_mean)/elev_std], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred = model(img1, img2, img3, e1, e2).item()
        pred_ft = pred * elev_std + elev_mean
        preds.append(pred_ft)
        targets.append(q['gage_height_ft'])
        sites_list.append(site)
        if len(samples) < 6:
            samples.append({'ref1': r1['image_path'], 'ref2': r2['image_path'], 'query': q['image_path'],
                           'e1': r1['gage_height_ft'], 'e2': r2['gage_height_ft'], 'true': q['gage_height_ft'], 'pred': pred_ft})

preds, targets = np.array(preds), np.array(targets)
mae = np.mean(np.abs(preds - targets))
rmse = np.sqrt(np.mean((preds - targets)**2))
ss_res = np.sum((targets - preds)**2)
ss_tot = np.sum((targets - np.mean(targets))**2)
r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

print(f'\n=== RESULTS ===\nMAE: {mae:.3f} ft ({mae*12:.1f} in)\nRMSE: {rmse:.3f} ft\nR²: {r2:.3f}')

# Plot 1: Predictions scatter
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(targets, preds, alpha=0.5, s=30, c='#2196F3')
mn, mx = min(targets.min(), preds.min())-0.5, max(targets.max(), preds.max())+0.5
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
ax.set_xlabel('Actual (ft)', fontsize=12)
ax.set_ylabel('Predicted (ft)', fontsize=12)
ax.set_title(f'Triplet RGB - MAE: {mae:.3f} ft, R²: {r2:.3f}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
plt.tight_layout()
plt.savefig(output_dir / 'test_predictions.png', dpi=150)
print(f'Saved: {output_dir}/test_predictions.png')

# Plot 2: Sample triplets
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, s in enumerate(samples[:6]):
    ax = axes[idx//3, idx%3]
    img = Image.open(s['query']).resize((224, 224))
    ax.imshow(img)
    ax.set_title(f"Ref1:{s['e1']:.1f} Ref2:{s['e2']:.1f}\nTrue:{s['true']:.2f} Pred:{s['pred']:.2f}", fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'sample_predictions.png', dpi=150)
print(f'Saved: {output_dir}/sample_predictions.png')

# Plot 3: Error by site
fig, ax = plt.subplots(figsize=(10, 6))
site_errors = {}
for p, t, s in zip(preds, targets, sites_list):
    site_errors.setdefault(s, []).append(abs(p-t))
site_mae = {s: np.mean(e) for s, e in site_errors.items()}
ax.bar(range(len(site_mae)), list(site_mae.values()))
ax.set_xticks(range(len(site_mae)))
ax.set_xticklabels([s[:20] for s in site_mae.keys()], rotation=45, ha='right')
ax.set_ylabel('MAE (ft)')
ax.set_title('Error by Test Site')
plt.tight_layout()
plt.savefig(output_dir / 'error_by_site.png', dpi=150)
print(f'Saved: {output_dir}/error_by_site.png')

# Save results
results = {'model': 'triplet_rgb', 'test_mae_ft': float(mae), 'test_mae_inches': float(mae*12),
           'test_rmse_ft': float(rmse), 'test_r2': float(r2), 'n_test_samples': len(preds),
           'test_sites': list(site_groups.keys()), 'elev_mean': float(elev_mean), 'elev_std': float(elev_std)}
with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {output_dir}/results.json')
print('\nDone!')
