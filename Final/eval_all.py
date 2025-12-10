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

class TripletRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.cross_attn = nn.MultiheadAttention(192, 4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(192*2+64, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, r1, r2, q, e1, e2):
        f1, f2, fq = self.backbone(r1), self.backbone(r2), self.backbone(q)
        attn, _ = self.cross_attn(fq.unsqueeze(1), torch.stack([f1,f2],1), torch.stack([f1,f2],1))
        return self.head(torch.cat([fq, attn.squeeze(1), self.elev_embed(torch.stack([e1,e2],1))], 1)).squeeze(-1)

class TripletSAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        old = base.patch_embed.proj
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight[:,:3] = old.weight
            new.weight[:,3:] = old.weight[:,:1]
            new.bias = old.bias
        base.patch_embed.proj = new
        self.backbone = base
        self.cross_attn = nn.MultiheadAttention(192, 4, dropout=0.1, batch_first=True)
        self.elev_embed = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(192*2+64, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, r1, r2, q, e1, e2):
        f1, f2, fq = self.backbone(r1), self.backbone(r2), self.backbone(q)
        attn, _ = self.cross_attn(fq.unsqueeze(1), torch.stack([f1,f2],1), torch.stack([f1,f2],1))
        return self.head(torch.cat([fq, attn.squeeze(1), self.elev_embed(torch.stack([e1,e2],1))], 1)).squeeze(-1)

def evaluate(model_dir, model_class, channels=3):
    print(f'\n{"="*50}\nEvaluating: {model_dir}\n{"="*50}')
    cp = torch.load(model_dir/'best_model.pt', weights_only=False)
    em, es = cp['elev_mean'], cp['elev_std']
    
    model = model_class()
    model.load_state_dict(cp['model_state_dict'])
    model.to(DEVICE).eval()
    
    df = pd.read_csv('data/test_quality.csv')
    sites = {s: g.reset_index(drop=True) for s, g in df.groupby('camera_id') if len(g)>=3}
    
    transform3 = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    preds, targs = [], []
    for site, sdf in tqdm(sites.items()):
        n = len(sdf)
        for _ in range(min(50, n)):
            i,j,k = np.random.choice(n, 3, replace=False)
            r1,r2,q = sdf.iloc[i], sdf.iloc[j], sdf.iloc[k]
            
            def load_img(path):
                img = Image.open(path).convert('RGB')
                t = transform3(img)
                if channels == 4:
                    mask = torch.ones(1, 224, 224) * 0.5
                    t = torch.cat([t, mask], 0)
                return t.unsqueeze(0).to(DEVICE)
            
            img1, img2, img3 = load_img(r1['image_path']), load_img(r2['image_path']), load_img(q['image_path'])
            e1 = torch.tensor([(r1['gage_height_ft']-em)/es], dtype=torch.float32).to(DEVICE)
            e2 = torch.tensor([(r2['gage_height_ft']-em)/es], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                pred = model(img1, img2, img3, e1, e2).item() * es + em
            preds.append(pred)
            targs.append(q['gage_height_ft'])
    
    preds, targs = np.array(preds), np.array(targs)
    mae = np.mean(np.abs(preds-targs))
    rmse = np.sqrt(np.mean((preds-targs)**2))
    r2 = 1 - np.sum((targs-preds)**2)/np.sum((targs-np.mean(targs))**2)
    
    print(f'MAE: {mae:.3f} ft ({mae*12:.1f} in)')
    print(f'RMSE: {rmse:.3f} ft')
    print(f'R2: {r2:.3f}')
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(targs, preds, alpha=0.5)
    mn, mx = min(targs.min(),preds.min())-0.5, max(targs.max(),preds.max())+0.5
    ax.plot([mn,mx],[mn,mx],'r--')
    ax.set_xlabel('Actual (ft)')
    ax.set_ylabel('Predicted (ft)')
    ax.set_title(f'{model_dir.name}\nMAE:{mae:.2f}ft R2:{r2:.3f}')
    plt.savefig(model_dir/'test_predictions.png', dpi=150)
    plt.close()
    
    with open(model_dir/'results.json','w') as f:
        json.dump({'mae_ft':float(mae),'rmse_ft':float(rmse),'r2':float(r2)},f,indent=2)
    
    return mae, r2

results = {}
for d in sorted(Path('outputs').glob('triplet_rgb_*')):
    if (d/'best_model.pt').exists():
        mae, r2 = evaluate(d, TripletRGB, 3)
        results[d.name] = {'mae': mae, 'r2': r2}

for d in sorted(Path('outputs').glob('triplet_sam_*')):
    if (d/'best_model.pt').exists():
        mae, r2 = evaluate(d, TripletSAM, 4)
        results[d.name] = {'mae': mae, 'r2': r2}

print('\n' + '='*50)
print('SUMMARY')
print('='*50)
for name, r in results.items():
    print(f"{name}: MAE={r['mae']:.3f}ft, R2={r['r2']:.3f}")
