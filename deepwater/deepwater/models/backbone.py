"""
Backbone Models for Water Level Estimation

This module provides Vision Transformer (ViT) and other backbone architectures
for extracting features from river/stream images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

logger = logging.getLogger(__name__)


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for image feature extraction.
    
    Uses pre-trained ViT models from timm library.
    """
    
    def __init__(
        self,
        model_name: str = "vit_large_patch16_224",
        pretrained: bool = True,
        freeze_layers: int = 0,
        output_dim: int = 1024,
    ):
        """
        Initialize ViT backbone.
        
        Args:
            model_name: timm model name
            pretrained: Whether to load pretrained weights
            freeze_layers: Number of transformer layers to freeze (from bottom)
            output_dim: Output feature dimension
        """
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm library required. Install with: pip install timm")
        
        # Load pretrained model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.model.num_features
        self.output_dim = output_dim
        
        # Projection layer if dimensions don't match
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze early layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        logger.info(
            f"Initialized {model_name} backbone: "
            f"feature_dim={self.feature_dim}, output_dim={output_dim}"
        )
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first N transformer layers."""
        # Freeze patch embedding
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze position embedding
        if hasattr(self.model, "pos_embed"):
            self.model.pos_embed.requires_grad = False
        
        # Freeze early blocks
        if hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        
        logger.info(f"Froze {num_layers} transformer layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, output_dim]
        """
        features = self.model(x)
        features = self.projection(features)
        return features
    
    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: List[int],
    ) -> List[torch.Tensor]:
        """
        Get intermediate features from specific layers.
        
        Args:
            x: Input images [B, C, H, W]
            layer_indices: Which layers to extract features from
            
        Returns:
            List of feature tensors
        """
        features = []
        
        # Get patch embeddings
        x = self.model.patch_embed(x)
        
        # Add position embedding
        if hasattr(self.model, "pos_embed"):
            x = x + self.model.pos_embed
        
        # Add CLS token
        if hasattr(self.model, "cls_token"):
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        # Pass through blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in layer_indices:
                # Extract CLS token features
                features.append(x[:, 0])
        
        return features


class CNNBackbone(nn.Module):
    """
    CNN backbone (ResNet) for image feature extraction.
    
    Alternative to ViT for environments with limited memory.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 768,
    ):
        """
        Initialize CNN backbone.
        
        Args:
            model_name: timm model name (resnet50, resnet101, etc.)
            pretrained: Whether to load pretrained weights
            output_dim: Output feature dimension
        """
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm library required")
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        self.feature_dim = self.model.num_features
        self.output_dim = output_dim
        
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        features = self.model(x)
        features = self.projection(features)
        return features


class FeatureFusion(nn.Module):
    """
    Module for fusing features from multiple images.
    
    Supports different fusion strategies:
    - concatenation
    - subtraction (for pairs)
    - attention-based fusion
    """
    
    def __init__(
        self,
        feature_dim: int,
        fusion_type: str = "concat",
        num_inputs: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize feature fusion.
        
        Args:
            feature_dim: Dimension of input features
            fusion_type: "concat", "subtract", or "attention"
            num_inputs: Number of input features to fuse
            hidden_dim: Hidden dimension for attention fusion
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.num_inputs = num_inputs
        
        if fusion_type == "concat":
            self.output_dim = feature_dim * num_inputs
        elif fusion_type == "subtract":
            assert num_inputs == 2, "Subtraction requires exactly 2 inputs"
            self.output_dim = feature_dim
        elif fusion_type == "attention":
            hidden_dim = hidden_dim or feature_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True,
            )
            self.output_dim = feature_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple feature vectors.
        
        Args:
            features: List of feature tensors [B, D]
            
        Returns:
            Fused features [B, output_dim]
        """
        if self.fusion_type == "concat":
            return torch.cat(features, dim=-1)
        
        elif self.fusion_type == "subtract":
            return features[1] - features[0]
        
        elif self.fusion_type == "attention":
            # Stack features as sequence
            stacked = torch.stack(features, dim=1)  # [B, N, D]
            
            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # Mean pool
            return attended.mean(dim=1)
        
        raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class WaterLevelHead(nn.Module):
    """
    Prediction head for water level estimation.
    
    Takes fused features and outputs water level predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_type: str = "regression",
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        """
        Initialize prediction head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_type: "regression" or "classification"
            num_classes: Number of classes for classification
            dropout: Dropout rate
        """
        super().__init__()
        
        self.output_type = output_type
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if output_type == "regression":
            self.output = nn.Linear(hidden_dim // 2, 1)
        else:
            self.output = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, input_dim]
            
        Returns:
            Predictions [B, 1] for regression or [B, num_classes] for classification
        """
        x = self.head(x)
        x = self.output(x)
        
        if self.output_type == "regression":
            x = x.squeeze(-1)
        
        return x


def create_backbone(
    backbone_type: str = "vit",
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    output_dim: int = 768,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create backbone models.
    
    Args:
        backbone_type: "vit" or "cnn"
        model_name: Specific model name
        pretrained: Whether to use pretrained weights
        output_dim: Output feature dimension
        **kwargs: Additional arguments
        
    Returns:
        Backbone model
    """
    if backbone_type == "vit":
        return ViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            **kwargs,
        )
    elif backbone_type == "cnn":
        return CNNBackbone(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
