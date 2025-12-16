# DeepWater: Water Level Estimation from River Camera Images

**CS771 - Learning Based Methods in Computer Vision**  
**University of Wisconsin-Madison, Fall 2024**

**Team Members:** Keegan Johnson, Forrest Peterson, Elice Priyadarshini, Jeffrey Weisinger

---

## Project Overview

We built a deep learning system that estimates water levels in rivers and streams from camera images. The USGS has hundreds of cameras monitoring waterways across the US, and our goal is to predict the water level (gage height) just by looking at the image - without needing the physical sensor data.

This could be useful for:
- Backup monitoring when sensors fail
- Extending monitoring to locations with cameras but no sensors
- Quick visual assessment during flood events

## The Problem

The challenge is that water level isn't directly visible in an image - you can't just count pixels. Different cameras have different angles, distances, and scenes. We need a model that can learn the *relative* appearance of water at different levels and generalize across sites.

## Our Approach

We tried two architectures:

### 1. Siamese Model (2 images)
Give the model a reference image with a known water level, plus a query image. The model predicts the query's water level.

```
[Reference Image] ──┐
                    ├──► ViT Backbone ──► Fusion ──► Predict water level
[Query Image] ──────┘
       +
[Known reference water level]
```

### 2. Triplet Model (3 images)
Give the model TWO reference images at different water levels, plus a query. This gives the model two "anchor points" to learn scale.

```
[Ref Image 1 + level] ──┐
[Ref Image 2 + level] ──┼──► ViT + Cross-Attention ──► Predict query level  
[Query Image] ──────────┘
```

Both models use a Vision Transformer (ViT) backbone pretrained on ImageNet.

## Data Collection

We collected data from 3 USGS camera sites:

| Camera Site | Location | Images | Water Level Range |
|-------------|----------|--------|-------------------|
| Yahara River at McFarland | Wisconsin | 150 | 4.48 - 7.15 ft |
| Mississippi River at Fridley | Minnesota | 150 | 2.29 - 3.51 ft |
| Allegheny River at Franklin | Pennsylvania | 149 | 5.63 - 8.15 ft |

**Total: 449 images** with synchronized water level measurements from USGS gauges.

The images were matched with gauge readings within 5 minutes of capture time.

## Results

### Model Comparison

| Model | MAE (ft) | MAE (inches) | RMSE (ft) | R² | Pearson r |
|-------|----------|--------------|-----------|-----|-----------|
| **Siamese (2-image)** | **0.073** | **~0.9"** | 0.142 | 0.994 | 0.997 |
| Triplet (3-image) | 0.108 | ~1.3" | 0.160 | 0.993 | 0.996 |

Both models achieve **sub-2-inch accuracy** with R² > 0.99!

### Siamese Model Performance

The Siamese model achieved the best results with an average error of less than 1 inch:

![Siamese Model Predictions](/cs771-project/outputs/siamese_20251202_102942/predictions.png)

### Triplet Model Performance

The Triplet model also performed well, though slightly behind Siamese on this dataset:

![Triplet Model Predictions](/cs771-project/outputs/triplet_20251202_140534/triplet_predictions.png)

### Why Siamese Won

Interestingly, the simpler Siamese model outperformed the more complex Triplet model. We think this is because:

1. **Limited data** - With only 449 images, the Triplet model's extra parameters were harder to train
2. **Consistent gauge data** - Our 3 sites have reliable ground truth, so we didn't need the "two anchor points" approach
3. **Faster convergence** - Siamese trained in ~8 min vs ~100 min for Triplet

The Triplet approach might shine more with cameras that have unknown or varying scales.

## How to Run

### Setup

```bash
# Clone the repo
git clone https://github.com/kejohnson32/cs771-project.git
cd cs771-project

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e packages/pynims
pip install -e deepwater
```

### Collect Data

```bash
# This downloads images and gauge data from USGS
python collect_training_data.py
```

### Train Siamese Model

```bash
python train_model.py
```

### Train Triplet Model

```bash
python train_triplet.py
```

### Evaluate Models

```bash
python eval_triplet.py
```

## Project Structure

```
cs771-project/
├── data/                          # Downloaded images and CSVs
│   ├── combined_dataset.csv       # All data merged
│   ├── WI_Yahara_River.../        # Camera-specific folders
│   │   ├── images/                # Downloaded images
│   │   └── gauge_data.csv         # Water level readings
│   └── ...
├── deepwater/                     # Our ML package
│   └── deepwater/
│       ├── models/                # Siamese & Triplet architectures
│       ├── data/                  # Dataset classes
│       ├── training/              # Training loop
│       └── utils/                 # Metrics, plotting
├── outputs/                       # Trained models and results
├── notebooks/                     # Existing project notebooks
├── packages/pynims/               # USGS API client
├── collect_training_data.py       # Data collection script
├── train_model.py                 # Siamese training
└── train_triplet.py               # Triplet training
```

## What We Learned

1. **Data synchronization matters** - Matching images to gauge readings within tight time windows (5 min) was crucial for clean labels.

2. **Pretrained ViT works well** - Even the tiny ViT model (5.6M params) achieved excellent results, showing transfer learning from ImageNet helps.

3. **Simpler can be better** - The 2-image Siamese model beat the 3-image Triplet model, likely due to limited training data.

4. **Normalization is important** - Water levels vary a lot between sites (2-8 ft). Normalizing targets globally helps training stability.

5. **M1 Macs are good** - Training on Apple Silicon with MPS was fast and worked out of the box with PyTorch.

## Limitations & Future Work

- **Only 3 camera sites** - Need to test generalization to unseen cameras
- **Limited water level variation** - Would benefit from flood event data
- **No seasonal variation** - All data from fall 2022/2024
- **Could add segmentation masks** - SAM2 water masks might help (we have notebooks for this)
- **Try larger ViT** - vit_small or vit_base might improve accuracy further

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- timm (for ViT models)
- pandas, numpy, matplotlib
- httpx (for API calls)
- dataretrieval (USGS data API)

## References

- USGS NIMS Camera System: https://apps.usgs.gov/hivis
- Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
- timm library: https://github.com/huggingface/pytorch-image-models

## Acknowledgments

Thanks to USGS for making the camera imagery and gauge data publicly available through their APIs. Also thanks to the CS771 course staff for guidance on this project.