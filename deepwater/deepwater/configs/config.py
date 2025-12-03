"""Configuration settings for DeepWater project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    # Directories
    data_root: str = "data"
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    
    # USGS API settings
    usgs_param_codes: dict = field(default_factory=lambda: {
        "00065": "gage_height_ft",      # Gage height in feet
        "00060": "discharge_cfs",        # Discharge in cubic feet per second
        "00045": "precipitation_in",     # Precipitation in inches
        "63160": "elevation_navd88_ft",  # Stream water level elevation (NAVD 88)
    })
    
    # Time sync settings
    max_time_diff_seconds: int = 60  # Max allowed time diff between image and gauge
    
    # Image settings
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    
    # Data quality
    min_images_per_site: int = 100  # Minimum images required for a site
    min_elevation_range_ft: float = 1.0  # Minimum water level variation required


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model type: "siamese" or "triplet"
    model_type: str = "siamese"
    
    # Backbone
    backbone: str = "vit_base_patch16_224"  # timm model name
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Input
    image_size: int = 224
    num_input_images: int = 2  # 2 for siamese, 3 for triplet
    
    # Architecture
    embed_dim: int = 768  # ViT base hidden dim
    projection_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    
    # Output
    output_type: str = "regression"  # "regression" or "classification"
    num_classes: int = 5  # Only used if output_type == "classification"


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Optimization
    optimizer: str = "adamw"
    gradient_clip: float = 1.0
    
    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    
    # Augmentation
    use_augmentation: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "deepwater"
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Mixed precision
    use_amp: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"


@dataclass 
class InferenceConfig:
    """Inference configuration."""
    
    model_path: str = ""
    batch_size: int = 32
    device: str = "auto"


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            inference=InferenceConfig(**config_dict.get("inference", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        config_dict = {
            "data": dataclasses.asdict(self.data),
            "model": dataclasses.asdict(self.model),
            "training": dataclasses.asdict(self.training),
            "inference": dataclasses.asdict(self.inference),
        }
        
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_device(device_str: str = "auto") -> str:
    """Get the appropriate device string."""
    import torch
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


# Default configuration
DEFAULT_CONFIG = Config()
