# DeepWater: Water Level Estimation from Camera Images

A computer vision system for estimating water levels from USGS monitoring camera images using deep learning.

## Authors
- Keegan Johnson (kejohnson32@wisc.edu)
- Forrest Peterson (fpeterson2@wisc.edu)
- Elice Priyadarshini (epriyadarshi@wisc.edu)
- Jeffrey Weisinger (jweisinger@wisc.edu)

University of Wisconsin-Madison | CS771 Machine Learning | Fall 2024

## Project Overview

We compare two architectures for water level estimation:
- **Siamese Model**: Uses 1 reference image with known water level
- **Triplet Model**: Uses 2 reference images with known water levels (enables scale calibration)

### Key Results

| Model | Seen Sites (MAE) | Seen R² | Unseen Sites (MAE) | Unseen R² |
|-------|------------------|---------|-------------------|-----------|
| Siamese RGB | 0.21 ft | 0.988 | 1.74 ft | -0.10 |
| Triplet RGB | 0.13 ft | 0.995 | 0.15 ft | 0.981 |

**Key Finding**: Triplet achieves 10x better generalization to unseen camera sites.

## Installation

```bash
# Clone repository
git clone https://github.com/kejohnson32/cs771-project.git
cd cs771-project

# Install dependencies
pip install -r requirements.txt

# (Optional) For SAM models, install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2 && pip install -e . && cd ..
```

## Data

### Download Data
```bash
python scripts/collect_quality_data.py
```

This downloads images from 21 USGS camera sites and creates:
- `data/quality_dataset.csv` - Full dataset (5,035 images)
- `data/train_quality.csv` - Training set (14 sites)
- `data/val_quality.csv` - Validation set (3 sites)
- `data/test_quality.csv` - Test set (4 unseen sites)

### Data Splits

**Cross-site split** (for true seen vs unseen evaluation):
- Train: 14 sites (seen during training)
- Val: 3 sites
- Test: 4 sites (completely unseen)

## Training

### Siamese Models
```bash
# Siamese RGB (3 channels)
python scripts/train_siamese_rgb.py --epochs 15

# Siamese SAM (4 channels with water mask)
python scripts/train_siamese_sam.py --epochs 15
```

### Triplet Models
```bash
# Triplet RGB (3 channels) - RECOMMENDED
python scripts/train_triplet_rgb.py --epochs 15

# Triplet SAM (4 channels with water mask)
python scripts/train_triplet_sam.py --epochs 15
```

## Evaluation

```bash
# Evaluate all models on seen vs unseen sites
python scripts/eval_seen_unseen.py
```

## Pre-trained Models

Pre-trained models are available in `models/`:
- `models/siamese_rgb/best_model.pt`
- `models/siamese_sam/best_model.pt`
- `models/triplet_rgb/best_model.pt`
- `models/triplet_sam/best_model.pt`

## Project Structure

```
cs771-project/
├── README.md
├── requirements.txt
├── scripts/
│   ├── collect_quality_data.py    # Data collection from USGS
│   ├── train_siamese_rgb.py       # Siamese model (RGB)
│   ├── train_siamese_sam.py       # Siamese model (RGB + SAM mask)
│   ├── train_triplet_rgb.py       # Triplet model (RGB)
│   ├── train_triplet_sam.py       # Triplet model (RGB + SAM mask)
│   └── eval_seen_unseen.py        # Evaluation script
├── models/
│   ├── siamese_rgb/
│   ├── siamese_sam/
│   ├── triplet_rgb/
│   └── triplet_sam/
├── data/
│   ├── quality_dataset.csv
│   ├── train_quality.csv
│   ├── val_quality.csv
│   └── test_quality.csv
└── outputs/                        # Training outputs and plots
```

## Architecture

### Siamese Model
```
Image 1 (Reference) ──┐
                      ├──> ViT Backbone ──> CAT ──> MLP Head ──> Prediction
Image 2 (Query) ──────┘                      ↑
                                             │
Water Level 1 ──> Elevation Embedding ───────┘
```

### Triplet Model
```
Image 1 (Ref 1) ──┐
Image 2 (Ref 2) ──┼──> ViT Backbone ──> Cross-Attention ──> CAT ──> MLP ──> Prediction
Image 3 (Query) ──┘         ↑                                  ↑
                            │                                  │
Water Level 1 & 2 ──> Elevation Embedding ─────────────────────┘
```

## Citation

```bibtex
@misc{deepwater2024,
  title={Water Level Estimation from Imagery Using Computer Vision},
  author={Johnson, Keegan and Peterson, Forrest and Priyadarshini, Elice and Weisinger, Jeffrey},
  year={2024},
  institution={University of Wisconsin-Madison}
}
```

## Acknowledgments

- USGS National Imagery Management System (NIMS) for camera data
- USGS Water Data for the Nation for gage height measurements
