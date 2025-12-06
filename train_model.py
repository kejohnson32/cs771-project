#!/usr/bin/env python3
"""
Train water level estimation model on M1 Mac.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Check device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using M1 GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🎮 Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")    
    print(f"💻 Using CPU")

# Import deepwater
# from deepwater.data import WaterLevelDataset, split_dataset, create_dataloaders
from deepwater.models import create_model
from deepwater.training import WaterLevelTrainer, create_optimizer, create_scheduler
from deepwater.utils import evaluate_model, print_metrics, plot_predictions, count_parameters, format_parameters

# Configuration
CONFIG = {
    'data_csv': 'data/combined_dataset.csv',
    'model_type': 'siamese',
    'backbone': 'vit_tiny_patch16_224',  # Small model for quick testing
    'batch_size': 8,  # Small batch for M1
    'num_epochs': 20,  # Quick test
    'learning_rate': 1e-4,
    'image_size': 224,
}

def get_images_dir(camera_id):
    """Get images directory for a camera."""
    return f'data/{camera_id}/images'

def main():
    print('='*60)
    print('Water Level Estimation Training')
    print('='*60)
    
    # Step 1: Load and check data
    print('\n1. Loading data...')
    df = pd.read_csv(CONFIG['data_csv'])
    print(f'   Total samples: {len(df)}')
    print(f'   Cameras: {df["camera_id"].unique().tolist()}')
    
    # Check elevation stats
    print(f'   Elevation range: {df["00065"].min():.2f} - {df["00065"].max():.2f} ft')
    print(f'   Elevation std: {df["00065"].std():.2f} ft')
    
    # Step 2: Prepare data - need to fix image paths
    print('\n2. Preparing dataset...')
    
    # The dataset expects a single images_dir, but we have multiple cameras
    # Let's create a unified structure
    
    # Update image paths to be full paths
    df['image_path'] = df.apply(
        lambda row: f"data/{row['camera_id']}/images/{row['image_name']}", 
        axis=1
    )
    
    # Check that images exist
    existing = df['image_path'].apply(os.path.exists)
    print(f'   Images found: {existing.sum()}/{len(df)}')
    
    if existing.sum() < 50:
        print('   ERROR: Not enough images found!')
        return
    
    # Filter to existing images only
    df = df[existing].reset_index(drop=True)
    
    # Rename columns for dataset compatibility
    df = df.rename(columns={
        'image_path': 'image_paths',
        'timestamp': 'image_times',
        '00065': 'gage_height_ft',
    })
    
    # Save processed data
    processed_csv = 'data/processed_dataset.csv'
    df.to_csv(processed_csv, index=False)
    print(f'   Saved processed data to {processed_csv}')
    
    # Step 3: Split dataset
    print('\n3. Splitting dataset...')
    
    # Manual split since our split_dataset expects different structure
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f'   Train: {len(train_df)} samples')
    print(f'   Val: {len(val_df)} samples')
    print(f'   Test: {len(test_df)} samples')
    
    # Step 4: Create custom dataset
    print('\n4. Creating dataloaders...')
    
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as T
    
    class SimpleWaterDataset(Dataset):
        def __init__(self, csv_path, transform=None):
            self.df = pd.read_csv(csv_path)
            self.transform = transform or T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Compute normalization stats
            self.elev_mean = self.df['gage_height_ft'].mean()
            self.elev_std = self.df['gage_height_ft'].std()
            
            # Pre-compute valid pairs (different elevation)
            self.pairs = []
            for i in range(len(self.df)):
                for j in range(len(self.df)):
                    if i != j:
                        elev_diff = abs(self.df.iloc[i]['gage_height_ft'] - self.df.iloc[j]['gage_height_ft'])
                        if elev_diff > 0.1:  # At least 0.1 ft difference
                            self.pairs.append((i, j))
            
            # Limit pairs for training efficiency
            if len(self.pairs) > 5000:
                np.random.shuffle(self.pairs)
                self.pairs = self.pairs[:5000]
        
        def __len__(self):
            return len(self.pairs)
        
        def normalize_elev(self, elev):
            return (elev - self.elev_mean) / self.elev_std
        
        def denormalize_elev(self, norm_elev):
            return norm_elev * self.elev_std + self.elev_mean
        
        def __getitem__(self, idx):
            i, j = self.pairs[idx]
            
            row1 = self.df.iloc[i]
            row2 = self.df.iloc[j]
            
            # Load images
            img1 = Image.open(row1['image_paths']).convert('RGB')
            img2 = Image.open(row2['image_paths']).convert('RGB')
            
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            # Get elevations (normalized)
            elev1 = self.normalize_elev(row1['gage_height_ft'])
            elev2 = self.normalize_elev(row2['gage_height_ft'])
            
            return {
                'image1': img1,
                'image2': img2,
                'elevation1': torch.tensor(elev1, dtype=torch.float32),
                'elevation2': torch.tensor(elev2, dtype=torch.float32),
            }
    
    # Create datasets
    train_dataset = SimpleWaterDataset('data/train.csv')
    val_dataset = SimpleWaterDataset('data/val.csv')
    test_dataset = SimpleWaterDataset('data/test.csv')
    
    # Copy normalization stats
    val_dataset.elev_mean = train_dataset.elev_mean
    val_dataset.elev_std = train_dataset.elev_std
    test_dataset.elev_mean = train_dataset.elev_mean
    test_dataset.elev_std = train_dataset.elev_std
    
    print(f'   Train pairs: {len(train_dataset)}')
    print(f'   Val pairs: {len(val_dataset)}')
    print(f'   Test pairs: {len(test_dataset)}')
    print(f'   Normalization: mean={train_dataset.elev_mean:.2f}, std={train_dataset.elev_std:.2f}')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Step 5: Create model
    print('\n5. Creating model...')
    
    model = create_model(
        model_type=CONFIG['model_type'],
        backbone=CONFIG['backbone'],
        pretrained=True,
        feature_dim=192,  # tiny ViT
        hidden_dim=128,
        dropout=0.1,
    )
    
    params = count_parameters(model)
    print(f'   Model: {CONFIG["model_type"]} with {CONFIG["backbone"]}')
    print(f'   Parameters: {format_parameters(params["total"])}')
    
    # Step 6: Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate=CONFIG['learning_rate'])
    scheduler = create_scheduler(optimizer, num_epochs=CONFIG['num_epochs'], warmup_epochs=2)
    
    # Step 7: Train
    print('\n6. Starting training...')
    print(f'   Device: {DEVICE}')
    print(f'   Epochs: {CONFIG["num_epochs"]}')
    print(f'   Batch size: {CONFIG["batch_size"]}')
    
    experiment_name = f"siamese_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trainer = WaterLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        output_dir='outputs',
        experiment_name=experiment_name,
        use_amp=False,  # MPS doesn't support AMP well yet
        early_stopping_patience=10,
    )
    
    history = trainer.train(num_epochs=CONFIG['num_epochs'])
    
    # Step 8: Evaluate
    print('\n7. Evaluating on test set...')
    
    # Load best model
    best_model_path = f'outputs/{experiment_name}/best_model.pt'
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            img1 = batch['image1'].to(DEVICE)
            img2 = batch['image2'].to(DEVICE)
            elev1 = batch['elevation1'].to(DEVICE)
            targets = batch['elevation2'].to(DEVICE)
            
            preds = model(img1, img2, elev1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Denormalize
    all_preds = np.array(all_preds) * train_dataset.elev_std + train_dataset.elev_mean
    all_targets = np.array(all_targets) * train_dataset.elev_std + train_dataset.elev_mean
    
    # Compute metrics
    from deepwater.utils import compute_all_metrics
    metrics = compute_all_metrics(all_preds, all_targets)
    print_metrics(metrics, title='Test Set Results')
    
    # Plot
    os.makedirs(f'outputs/{experiment_name}', exist_ok=True)
    plot_predictions(
        all_preds, all_targets,
        title='Water Level Predictions',
        save_path=f'outputs/{experiment_name}/predictions.png'
    )
    print(f'\n   Plot saved to outputs/{experiment_name}/predictions.png')
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE!')
    print('='*60)
    print(f'Best model: outputs/{experiment_name}/best_model.pt')
    print(f'MAE: {metrics["mae"]:.3f} ft')
    print(f'RMSE: {metrics["rmse"]:.3f} ft')


if __name__ == '__main__':
    main()