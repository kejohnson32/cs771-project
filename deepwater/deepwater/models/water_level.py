"""
Water Level Estimation Models

This module provides the main model architectures for water level estimation:
- SiameseWaterLevelModel: Uses 2 images (reference + query)
- TripletWaterLevelModel: Uses 3 images (2 references + query)

The models learn to predict the water level of a query image given
reference images with known water levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import logging

from .backbone import (
    ViTBackbone,
    CNNBackbone,
    FeatureFusion,
    WaterLevelHead,
    create_backbone,
)

logger = logging.getLogger(__name__)


class SiameseWaterLevelModel(nn.Module):
    """
    Siamese network for water level estimation.
    
    Architecture:
    1. Shared backbone extracts features from both images
    2. Features are fused (concatenation, subtraction, or attention)
    3. Elevation values are embedded and concatenated
    4. Prediction head outputs water level estimate
    
    The model learns to predict the water level of image2 given:
    - Image1 (reference) with known elevation1
    - Image2 (query)
    """
    
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        fusion_type: str = "concat",
        output_type: str = "regression",
        num_classes: int = 5,
        freeze_backbone_layers: int = 0,
        use_elevation_embedding: bool = True,
        elevation_embed_dim: int = 64,
    ):
        """
        Initialize Siamese model.
        
        Args:
            backbone: Backbone model name
            pretrained: Use pretrained weights
            feature_dim: Feature dimension from backbone
            hidden_dim: Hidden dimension for prediction head
            dropout: Dropout rate
            fusion_type: How to fuse features ("concat", "subtract", "attention")
            output_type: "regression" or "classification"
            num_classes: Number of classes for classification
            freeze_backbone_layers: Number of backbone layers to freeze
            use_elevation_embedding: Whether to embed elevation values
            elevation_embed_dim: Dimension of elevation embedding
        """
        super().__init__()
        
        self.use_elevation_embedding = use_elevation_embedding
        self.output_type = output_type
        
        # Create shared backbone
        backbone_type = "vit" if "vit" in backbone.lower() else "cnn"
        self.backbone = create_backbone(
            backbone_type=backbone_type,
            model_name=backbone,
            pretrained=pretrained,
            output_dim=feature_dim,
            freeze_layers=freeze_backbone_layers,
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            num_inputs=2,
        )
        
        # Elevation embedding
        if use_elevation_embedding:
            self.elevation_embed = nn.Sequential(
                nn.Linear(1, elevation_embed_dim),
                nn.LayerNorm(elevation_embed_dim),
                nn.GELU(),
            )
            head_input_dim = self.fusion.output_dim + elevation_embed_dim
        else:
            head_input_dim = self.fusion.output_dim + 1  # Just concatenate raw value
        
        # Prediction head
        self.head = WaterLevelHead(
            input_dim=head_input_dim,
            hidden_dim=hidden_dim,
            output_type=output_type,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        logger.info(
            f"SiameseWaterLevelModel initialized: "
            f"backbone={backbone}, fusion={fusion_type}, output={output_type}"
        )
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        elevation1: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            image1: Reference image [B, C, H, W]
            image2: Query image [B, C, H, W]
            elevation1: Known elevation of reference image [B]
            return_features: Whether to return intermediate features
            
        Returns:
            Predicted elevation for image2 [B] (or [B, num_classes] for classification)
            Optionally also returns fused features
        """
        # Extract features
        feat1 = self.backbone(image1)
        feat2 = self.backbone(image2)
        
        # Fuse features
        fused = self.fusion([feat1, feat2])
        
        # Process elevation
        elev = elevation1.unsqueeze(-1) if elevation1.dim() == 1 else elevation1
        
        if self.use_elevation_embedding:
            elev_embed = self.elevation_embed(elev)
            combined = torch.cat([fused, elev_embed], dim=-1)
        else:
            combined = torch.cat([fused, elev], dim=-1)
        
        # Predict
        output = self.head(combined)
        
        if return_features:
            return output, fused
        return output
    
    def predict_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict from a batch dictionary (from dataloader).
        
        Args:
            batch: Dictionary with "image1", "image2", "elevation1"
            
        Returns:
            Predictions
        """
        return self.forward(
            image1=batch["image1"],
            image2=batch["image2"],
            elevation1=batch["elevation1"],
        )


class TripletWaterLevelModel(nn.Module):
    """
    Triplet network for water level estimation.
    
    This model addresses the challenge of unknown absolute scale by using
    two reference images at different known water levels.
    
    Architecture:
    1. Shared backbone extracts features from all three images
    2. Cross-attention between query and references
    3. Elevation values are embedded
    4. Prediction head outputs water level estimate
    
    The model learns to predict the water level of the query image given:
    - Ref1 with known elevation1
    - Ref2 with known elevation2
    - Query image
    """
    
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_type: str = "regression",
        num_classes: int = 5,
        freeze_backbone_layers: int = 0,
        elevation_embed_dim: int = 64,
    ):
        """
        Initialize Triplet model.
        
        Args:
            backbone: Backbone model name
            pretrained: Use pretrained weights
            feature_dim: Feature dimension from backbone
            hidden_dim: Hidden dimension for prediction head
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_type: "regression" or "classification"
            num_classes: Number of classes for classification
            freeze_backbone_layers: Number of backbone layers to freeze
            elevation_embed_dim: Dimension of elevation embedding
        """
        super().__init__()
        
        self.output_type = output_type
        self.feature_dim = feature_dim
        
        # Create shared backbone
        backbone_type = "vit" if "vit" in backbone.lower() else "cnn"
        self.backbone = create_backbone(
            backbone_type=backbone_type,
            model_name=backbone,
            pretrained=pretrained,
            output_dim=feature_dim,
            freeze_layers=freeze_backbone_layers,
        )
        
        # Elevation embedding
        self.elevation_embed = nn.Sequential(
            nn.Linear(1, elevation_embed_dim),
            nn.LayerNorm(elevation_embed_dim),
            nn.GELU(),
        )
        
        # Feature projection (add elevation info to features)
        self.feature_proj = nn.Linear(feature_dim + elevation_embed_dim, feature_dim)
        
        # Cross-attention: query attends to references
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention for final processing
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # MLP for feature processing
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )
        
        # Prediction head
        self.head = WaterLevelHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_type=output_type,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        logger.info(
            f"TripletWaterLevelModel initialized: "
            f"backbone={backbone}, heads={num_heads}, output={output_type}"
        )
    
    def forward(
        self,
        ref1: torch.Tensor,
        ref2: torch.Tensor,
        query: torch.Tensor,
        elevation1: torch.Tensor,
        elevation2: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            ref1: First reference image [B, C, H, W]
            ref2: Second reference image [B, C, H, W]
            query: Query image [B, C, H, W]
            elevation1: Known elevation of ref1 [B]
            elevation2: Known elevation of ref2 [B]
            return_features: Whether to return intermediate features
            
        Returns:
            Predicted elevation for query [B]
        """
        batch_size = ref1.size(0)
        
        # Extract features
        feat1 = self.backbone(ref1)  # [B, D]
        feat2 = self.backbone(ref2)  # [B, D]
        feat_query = self.backbone(query)  # [B, D]
        
        # Embed elevations
        elev1 = elevation1.unsqueeze(-1) if elevation1.dim() == 1 else elevation1
        elev2 = elevation2.unsqueeze(-1) if elevation2.dim() == 1 else elevation2
        
        elev_embed1 = self.elevation_embed(elev1)  # [B, E]
        elev_embed2 = self.elevation_embed(elev2)  # [B, E]
        
        # Combine features with elevation info
        feat1_with_elev = torch.cat([feat1, elev_embed1], dim=-1)
        feat2_with_elev = torch.cat([feat2, elev_embed2], dim=-1)
        
        ref1_proj = self.feature_proj(feat1_with_elev)  # [B, D]
        ref2_proj = self.feature_proj(feat2_with_elev)  # [B, D]
        
        # Stack references as key-value sequence
        refs = torch.stack([ref1_proj, ref2_proj], dim=1)  # [B, 2, D]
        
        # Query attends to references
        query_seq = feat_query.unsqueeze(1)  # [B, 1, D]
        attended, _ = self.cross_attention(query_seq, refs, refs)  # [B, 1, D]
        
        # Residual connection and norm
        x = self.norm1(query_seq + attended)
        
        # Self-attention
        x_attn, _ = self.self_attention(x, x, x)
        x = self.norm2(x + x_attn)
        
        # MLP
        x = x + self.mlp(x)
        
        # Get final features
        features = x.squeeze(1)  # [B, D]
        
        # Predict
        output = self.head(features)
        
        if return_features:
            return output, features
        return output
    
    def predict_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict from a batch dictionary (from dataloader).
        
        Args:
            batch: Dictionary with "ref1", "ref2", "query", "elevation1", "elevation2"
            
        Returns:
            Predictions
        """
        return self.forward(
            ref1=batch["ref1"],
            ref2=batch["ref2"],
            query=batch["query"],
            elevation1=batch["elevation1"],
            elevation2=batch["elevation2"],
        )


class WaterLevelDifferenceModel(nn.Module):
    """
    Model that predicts the DIFFERENCE in water level between two images.
    
    This is simpler than predicting absolute water levels and can be useful
    for relative change detection.
    """
    
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize difference model.
        
        Args:
            backbone: Backbone model name
            pretrained: Use pretrained weights
            feature_dim: Feature dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create shared backbone
        backbone_type = "vit" if "vit" in backbone.lower() else "cnn"
        self.backbone = create_backbone(
            backbone_type=backbone_type,
            model_name=backbone,
            pretrained=pretrained,
            output_dim=feature_dim,
        )
        
        # Prediction head (from feature difference)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict water level difference.
        
        Args:
            image1: First image [B, C, H, W]
            image2: Second image [B, C, H, W]
            
        Returns:
            Predicted difference (elevation2 - elevation1) [B]
        """
        # Extract features
        feat1 = self.backbone(image1)
        feat2 = self.backbone(image2)
        
        # Compute difference
        diff = feat2 - feat1
        
        # Predict
        output = self.head(diff).squeeze(-1)
        
        return output


def create_model(
    model_type: str = "siamese",
    backbone: str = "vit_base_patch16_224",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create water level models.
    
    Args:
        model_type: "siamese", "triplet", or "difference"
        backbone: Backbone model name
        pretrained: Use pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    if model_type == "siamese":
        return SiameseWaterLevelModel(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs,
        )
    elif model_type == "triplet":
        return TripletWaterLevelModel(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs,
        )
    elif model_type == "difference":
        return WaterLevelDifferenceModel(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model size variants
MODEL_CONFIGS = {
    "siamese_tiny": {
        "model_type": "siamese",
        "backbone": "vit_tiny_patch16_224",
        "feature_dim": 192,
        "hidden_dim": 128,
    },
    "siamese_small": {
        "model_type": "siamese",
        "backbone": "vit_small_patch16_224",
        "feature_dim": 384,
        "hidden_dim": 192,
    },
    "siamese_base": {
        "model_type": "siamese",
        "backbone": "vit_base_patch16_224",
        "feature_dim": 768,
        "hidden_dim": 256,
    },
    "triplet_small": {
        "model_type": "triplet",
        "backbone": "vit_small_patch16_224",
        "feature_dim": 384,
        "hidden_dim": 192,
    },
    "triplet_base": {
        "model_type": "triplet",
        "backbone": "vit_base_patch16_224",
        "feature_dim": 768,
        "hidden_dim": 256,
    },
    # CNN alternatives (lower memory)
    "siamese_resnet50": {
        "model_type": "siamese",
        "backbone": "resnet50",
        "feature_dim": 768,
        "hidden_dim": 256,
    },
}


def create_model_from_config(config_name: str, **kwargs) -> nn.Module:
    """
    Create model from predefined configuration.
    
    Args:
        config_name: Name of configuration
        **kwargs: Override config values
        
    Returns:
        Model instance
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    return create_model(**config)
