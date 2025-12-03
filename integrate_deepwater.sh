#!/bin/bash
# DeepWater Integration Script
# Run this from inside your cs771-project directory
# Usage: ./integrate_deepwater.sh [milestone_number]
# Example: ./integrate_deepwater.sh 1

set -e

MILESTONE=${1:-0}

echo "================================================"
echo "DeepWater Integration - Milestone $MILESTONE"
echo "================================================"

case $MILESTONE in
    0)
        echo "Setting up branch and directory structure..."
        git checkout -b feature/deepwater-ml-pipeline 2>/dev/null || git checkout feature/deepwater-ml-pipeline
        mkdir -p deepwater/deepwater/{configs,data,models,training,utils}
        mkdir -p deepwater/scripts
        mkdir -p deepwater/notebooks
        mkdir -p deepwater/configs
        echo "Done! Now run: ./integrate_deepwater.sh 1"
        ;;
    
    1)
        echo "Milestone 1: Adding project configuration..."
        
        # Create pyproject.toml
        cat > deepwater/pyproject.toml << 'PYPROJECT'
[project]
name = "deepwater"
version = "0.1.0"
description = "Water Level Estimation from Imagery Using Computer Vision"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "timm>=0.9.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "pillow>=10.0.0",
    "opencv-python>=4.8.0",
    "scikit-image>=0.21.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.65.0",
    "httpx>=0.24.0",
    "pytz>=2023.3",
    "python-dateutil>=2.8.2",
    "dataretrieval>=1.0.0",
    "pyyaml>=6.0",
    "albumentations>=1.3.0",
]

[project.optional-dependencies]
wandb = ["wandb>=0.15.0"]
dev = ["ipykernel>=6.25.0", "pytest>=7.4.0"]

[project.scripts]
deepwater = "deepwater.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["deepwater"]
PYPROJECT

        # Create main __init__.py
        cat > deepwater/deepwater/__init__.py << 'INIT'
"""DeepWater - Water Level Estimation from Imagery Using Computer Vision."""
__version__ = "0.1.0"
INIT

        # Create configs/__init__.py
        cat > deepwater/deepwater/configs/__init__.py << 'CONFIGINIT'
from .config import (
    Config, DataConfig, ModelConfig, TrainingConfig,
    InferenceConfig, get_device, DEFAULT_CONFIG,
)
__all__ = ["Config", "DataConfig", "ModelConfig", "TrainingConfig",
           "InferenceConfig", "get_device", "DEFAULT_CONFIG"]
CONFIGINIT

        echo "Now copy config.py from the milestones/01_config folder"
        echo "Then run:"
        echo "  git add deepwater/"
        echo "  git commit -m 'feat: Add deepwater package structure and configuration'"
        echo "  ./integrate_deepwater.sh 2"
        ;;
    
    2)
        echo "Milestone 2: Adding site discovery module..."
        
        # Create data/__init__.py (basic version)
        cat > deepwater/deepwater/data/__init__.py << 'DATAINIT'
"""Data module for DeepWater."""
from .site_discovery import SiteDiscovery, SiteInfo, discover_best_sites
__all__ = ["SiteDiscovery", "SiteInfo", "discover_best_sites"]
DATAINIT

        echo "Now copy site_discovery.py from milestones/02_site_discovery/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/data/"
        echo "  git commit -m 'feat: Add site discovery module for ranking USGS camera sites'"
        echo "  ./integrate_deepwater.sh 3"
        ;;
    
    3)
        echo "Milestone 3: Adding dataset classes..."
        echo "Copy dataset.py from milestones/03_dataset/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/data/dataset.py"
        echo "  git commit -m 'feat: Add PyTorch dataset classes for Siamese and Triplet models'"
        echo "  ./integrate_deepwater.sh 4"
        ;;
    
    4)
        echo "Milestone 4: Adding data collection pipeline..."
        
        # Update data/__init__.py
        cat > deepwater/deepwater/data/__init__.py << 'DATAINIT'
"""Data module for DeepWater."""
from .dataset import (
    WaterLevelDataset, MultiSiteDataset, ImageSample,
    create_dataloaders, split_dataset,
)
from .collection import DataCollector, CollectionConfig, collect_training_data
from .site_discovery import SiteDiscovery, SiteInfo, discover_best_sites

__all__ = [
    "WaterLevelDataset", "MultiSiteDataset", "ImageSample",
    "create_dataloaders", "split_dataset",
    "DataCollector", "CollectionConfig", "collect_training_data",
    "SiteDiscovery", "SiteInfo", "discover_best_sites",
]
DATAINIT

        echo "Copy collection.py from milestones/04_collection/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/data/"
        echo "  git commit -m 'feat: Add data collection pipeline for downloading images and gauge data'"
        echo "  ./integrate_deepwater.sh 5"
        ;;
    
    5)
        echo "Milestone 5: Adding backbone models..."
        
        cat > deepwater/deepwater/models/__init__.py << 'MODELINIT'
"""Models module for DeepWater."""
from .backbone import (
    ViTBackbone, CNNBackbone, FeatureFusion,
    WaterLevelHead, create_backbone,
)
__all__ = ["ViTBackbone", "CNNBackbone", "FeatureFusion",
           "WaterLevelHead", "create_backbone"]
MODELINIT

        echo "Copy backbone.py from milestones/05_backbone/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/models/"
        echo "  git commit -m 'feat: Add ViT and CNN backbone models for feature extraction'"
        echo "  ./integrate_deepwater.sh 6"
        ;;
    
    6)
        echo "Milestone 6: Adding water level estimation models..."
        
        # Update models/__init__.py
        cat > deepwater/deepwater/models/__init__.py << 'MODELINIT'
"""Models module for DeepWater."""
from .backbone import (
    ViTBackbone, CNNBackbone, FeatureFusion,
    WaterLevelHead, create_backbone,
)
from .water_level import (
    SiameseWaterLevelModel, TripletWaterLevelModel,
    WaterLevelDifferenceModel, create_model,
    create_model_from_config, MODEL_CONFIGS,
)

__all__ = [
    "ViTBackbone", "CNNBackbone", "FeatureFusion",
    "WaterLevelHead", "create_backbone",
    "SiameseWaterLevelModel", "TripletWaterLevelModel",
    "WaterLevelDifferenceModel", "create_model",
    "create_model_from_config", "MODEL_CONFIGS",
]
MODELINIT

        echo "Copy water_level.py from milestones/06_models/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/models/"
        echo "  git commit -m 'feat: Add Siamese and Triplet models for water level estimation'"
        echo "  ./integrate_deepwater.sh 7"
        ;;
    
    7)
        echo "Milestone 7: Adding training module..."
        
        cat > deepwater/deepwater/training/__init__.py << 'TRAININIT'
"""Training module for DeepWater."""
from .trainer import (
    WaterLevelTrainer, TrainingMetrics, EarlyStopping,
    WarmupScheduler, create_optimizer, create_scheduler,
)
__all__ = ["WaterLevelTrainer", "TrainingMetrics", "EarlyStopping",
           "WarmupScheduler", "create_optimizer", "create_scheduler"]
TRAININIT

        echo "Copy trainer.py from milestones/07_training/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/training/"
        echo "  git commit -m 'feat: Add training module with trainer, optimizers, and schedulers'"
        echo "  ./integrate_deepwater.sh 8"
        ;;
    
    8)
        echo "Milestone 8: Adding utilities..."
        
        cat > deepwater/deepwater/utils/__init__.py << 'UTILINIT'
"""Utilities module for DeepWater."""
from .evaluation import (
    compute_mae, compute_rmse, compute_mape, compute_r2,
    compute_pearson_correlation, compute_all_metrics, print_metrics,
    evaluate_model, plot_predictions, plot_error_distribution,
    plot_training_history, estimate_dataset_size,
    count_parameters, format_parameters,
)
__all__ = [
    "compute_mae", "compute_rmse", "compute_mape", "compute_r2",
    "compute_pearson_correlation", "compute_all_metrics", "print_metrics",
    "evaluate_model", "plot_predictions", "plot_error_distribution",
    "plot_training_history", "estimate_dataset_size",
    "count_parameters", "format_parameters",
]
UTILINIT

        echo "Copy evaluation.py from milestones/08_utils/"
        echo "Then run:"
        echo "  git add deepwater/deepwater/utils/"
        echo "  git commit -m 'feat: Add evaluation metrics and visualization utilities'"
        echo "  ./integrate_deepwater.sh 9"
        ;;
    
    9)
        echo "Milestone 9: Adding CLI and scripts..."
        
        echo "Copy the following files:"
        echo "  - cli.py -> deepwater/deepwater/cli.py"
        echo "  - train.py -> deepwater/scripts/train.py"
        echo "  - default.yaml -> deepwater/configs/default.yaml"
        echo ""
        echo "Then run:"
        echo "  git add deepwater/deepwater/cli.py deepwater/scripts/ deepwater/configs/"
        echo "  git commit -m 'feat: Add CLI and training scripts'"
        echo "  ./integrate_deepwater.sh 10"
        ;;
    
    10)
        echo "Milestone 10: Adding documentation..."
        
        echo "Copy the following files:"
        echo "  - README.md -> deepwater/README.md"
        echo "  - train_colab.ipynb -> deepwater/notebooks/train_colab.ipynb"
        echo ""
        echo "Then run:"
        echo "  git add deepwater/README.md deepwater/notebooks/"
        echo "  git commit -m 'docs: Add README and Colab training notebook'"
        echo ""
        echo "Final steps:"
        echo "  git push -u origin feature/deepwater-ml-pipeline"
        echo "  # Create PR on GitHub"
        ;;
    
    test)
        echo "Testing installation..."
        cd deepwater
        pip install -e . --quiet
        pip install -e ../packages/pynims --quiet
        python -c "from deepwater.configs import Config; print('✓ Configs')"
        python -c "from deepwater.data import WaterLevelDataset; print('✓ Data')"
        python -c "from deepwater.models import create_model; print('✓ Models')"
        python -c "from deepwater.training import WaterLevelTrainer; print('✓ Training')"
        python -c "from deepwater.utils import compute_mae; print('✓ Utils')"
        echo ""
        echo "All imports successful! ✓"
        ;;
    
    *)
        echo "Unknown milestone: $MILESTONE"
        echo "Usage: ./integrate_deepwater.sh [0-10|test]"
        echo ""
        echo "Milestones:"
        echo "  0  - Setup branch and directory structure"
        echo "  1  - Project configuration"
        echo "  2  - Site discovery module"
        echo "  3  - Dataset classes"
        echo "  4  - Data collection pipeline"
        echo "  5  - Backbone models"
        echo "  6  - Water level models"
        echo "  7  - Training module"
        echo "  8  - Utilities"
        echo "  9  - CLI and scripts"
        echo "  10 - Documentation"
        echo "  test - Test the installation"
        ;;
esac
