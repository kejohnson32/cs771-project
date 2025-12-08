#!/usr/bin/env python3
"""
Train TRIPLET model - lighter version for M1 Mac
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import gc

# Force MPS to release memory more aggressively
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Check device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🍎 Using M1 GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA GPU!")
else:
    DEVICE = torch.device("cpu")
    print(f"💻 Using CPU")

from deepwater.models import create_model
from deepwater.utils import compute_all_metrics, print_metrics, count_parameters, format_parameters

# Configuration - SMALLER for M1
CONFIG = {
    'data_csv': 'data/combined_dataset.csv',
    'batch_size': 4,  # Smaller batch!
    'num_epochs': 20,
    'learning_rate': 1e-4,
}


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, elev_mean=None, elev_std=None, max_triplets=3000):
        import torchvision.transforms as T
        
        self.df = pd.read_csv(csv_path)
        print(f"Length before: {len(self.df)}")
        nan_count = self.df['gage_height_ft'].isna().sum()
        inf_count = np.isinf(self.df['gage_height_ft']).sum()    
        print(f"We have {nan_count} nan values")
        print(f"We have {inf_count} infinite values")
        self.df['gage_height_ft'] = pd.to_numeric(self.df['gage_height_ft'])
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna(subset=['gage_height_ft']).reset_index(drop=True)
        print(f"Length after: {len(self.df)}")
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if elev_mean is None:
            self.elev_mean = self.df['gage_height_ft'].mean()
            self.elev_std = self.df['gage_height_ft'].std()
        else:
            self.elev_mean = elev_mean
            self.elev_std = elev_std
        
        # Build triplets - FEWER than before
        self.triplets = []
        n = len(self.df)
        
        attempts = 0
        while len(self.triplets) < max_triplets and attempts < max_triplets * 10:
            i = np.random.randint(n)
            j = np.random.randint(n)
            k = np.random.randint(n)
            
            if i == j or j == k or i == k:
                attempts += 1
                continue
            
            elev_i = self.df.iloc[i]['gage_height_ft']
            elev_j = self.df.iloc[j]['gage_height_ft']
            
            if abs(elev_i - elev_j) >= 0.3:
                self.triplets.append((i, j, k))
            
            attempts += 1
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        i, j, k = self.triplets[idx]
        
        row1 = self.df.iloc[i]
        row2 = self.df.iloc[j]
        row3 = self.df.iloc[k]
        
        img1 = Image.open(row1['image_paths']).convert('RGB')
        img2 = Image.open(row2['image_paths']).convert('RGB')
        img3 = Image.open(row3['image_paths']).convert('RGB')
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)


        
        elev1 = (row1['gage_height_ft'] - self.elev_mean) / self.elev_std
        elev2 = (row2['gage_height_ft'] - self.elev_mean) / self.elev_std
        elev3 = (row3['gage_height_ft'] - self.elev_mean) / self.elev_std
        
        return {
            'ref1': img1, 'ref2': img2, 'query': img3,
            'elevation1': torch.tensor(elev1, dtype=torch.float32),
            'elevation2': torch.tensor(elev2, dtype=torch.float32),
            'elevation_query': torch.tensor(elev3, dtype=torch.float32),
        }



def main():
    print('='*60)
    print('TRIPLET Model Training (Lightweight)')
    print('='*60)
    
    # Load data
    print('\n1. Loading data...')
    df = pd.read_csv(CONFIG['data_csv'])
    df['image_paths'] = df.apply(lambda row: f"data/{row['camera_id']}/images/{row['image_name']}", axis=1)
    df = df[df['image_paths'].apply(os.path.exists)].reset_index(drop=True)
    df = df.rename(columns={'00065': 'gage_height_ft'})
    print(f'   Samples: {len(df)}')
    
    # Split
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv('data/train_triplet.csv', index=False)
    val_df.to_csv('data/val_triplet.csv', index=False)
    test_df.to_csv('data/test_triplet.csv', index=False)
    print(f'   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    # Datasets - smaller
    print('\n2. Creating datasets...')
    train_dataset = TripletDataset('data/train_triplet.csv', max_triplets=3000)
    val_dataset = TripletDataset('data/val_triplet.csv', train_dataset.elev_mean, train_dataset.elev_std, max_triplets=1000)
    test_dataset = TripletDataset('data/test_triplet.csv', train_dataset.elev_mean, train_dataset.elev_std, max_triplets=1000)
    
    print(f'   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    print('\n3. Creating model...')
    model = create_model(model_type='triplet', backbone='vit_tiny_patch16_224', pretrained=True, feature_dim=192, hidden_dim=128, dropout=0.1)
    model.to(DEVICE)
    print(f'   Parameters: {format_parameters(count_parameters(model)["total"])}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    # Training
    print('\n4. Training...')
    experiment_name = f"triplet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path('outputs') / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        model.train()
        train_loss, train_mae, n = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]}', leave=True)
        for batch in pbar:
            ref1 = batch['ref1'].to(DEVICE)
            ref2 = batch['ref2'].to(DEVICE)
            query = batch['query'].to(DEVICE)
            elev1 = batch['elevation1'].to(DEVICE)
            elev2 = batch['elevation2'].to(DEVICE)
            targets = batch['elevation_query'].to(DEVICE)
            
            optimizer.zero_grad()

            preds = model(ref1, ref2, query, elev1, elev2)

            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            bs = targets.size(0)
            train_loss += loss.item() * bs
            train_mae += F.l1_loss(preds, targets, reduction='sum').item()
            n += bs
            
            pbar.set_postfix({'loss': train_loss/n, 'mae_ft': (train_mae/n)*train_dataset.elev_std})
            
            # Clear cache periodically
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
        
        # Validate
        model.eval()
        val_loss, val_mae, n_val = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ref1 = batch['ref1'].to(DEVICE)
                ref2 = batch['ref2'].to(DEVICE)
                query = batch['query'].to(DEVICE)
                elev1 = batch['elevation1'].to(DEVICE)
                elev2 = batch['elevation2'].to(DEVICE)
                targets = batch['elevation_query'].to(DEVICE)
                
                preds = model(ref1, ref2, query, elev1, elev2)
                loss = F.mse_loss(preds, targets)
                
                bs = targets.size(0)
                val_loss += loss.item() * bs
                val_mae += F.l1_loss(preds, targets, reduction='sum').item()
                n_val += bs
        
        scheduler.step()
        
        train_mae_ft = (train_mae / n) * train_dataset.elev_std
        val_mae_ft = (val_mae / n_val) * train_dataset.elev_std
        
        print(f'   Val Loss: {val_loss/n_val:.4f} | Val MAE: {val_mae_ft:.3f}ft')
        
        if val_loss/n_val < best_val_loss:
            best_val_loss = val_loss/n_val
            torch.save({'model_state_dict': model.state_dict(), 'elev_mean': train_dataset.elev_mean, 'elev_std': train_dataset.elev_std}, output_dir / 'best_model.pt')
            print(f'   ✓ Saved best model!')
        
        # Clear memory
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
    
    # Evaluate
    print('\n5. Evaluating...')
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch['ref1'].to(DEVICE), batch['ref2'].to(DEVICE), batch['query'].to(DEVICE), batch['elevation1'].to(DEVICE), batch['elevation2'].to(DEVICE))
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch['elevation_query'].numpy())
    
    all_preds = np.array(all_preds) * train_dataset.elev_std + train_dataset.elev_mean
    all_targets = np.array(all_targets) * train_dataset.elev_std + train_dataset.elev_mean
    
    metrics = compute_all_metrics(all_preds, all_targets)
    print_metrics(metrics, title='TRIPLET Test Results')
    
    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_targets, all_preds, alpha=0.5, s=20, c='green')
    mn, mx = min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())
    ax.plot([mn, mx], [mn, mx], 'r--')
    ax.set_xlabel('Actual (ft)')
    ax.set_ylabel('Predicted (ft)')
    ax.set_title(f'TRIPLET: MAE={metrics["mae"]:.3f}ft, R²={metrics["r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'triplet_results.png', dpi=150)
    print(f'\nSaved to {output_dir}/')
    
    print('\n' + '='*60)
    print(f'TRIPLET MAE:  {metrics["mae"]:.3f} ft')
    print(f'TRIPLET RMSE: {metrics["rmse"]:.3f} ft')
    print(f'TRIPLET R²:   {metrics["r2"]:.3f}')
    print('='*60)


if __name__ == '__main__':
    main()
