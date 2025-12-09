#!/usr/bin/env python3
"""
Train Triplet model with SAM2 water masks (RGB + Mask = 4 channels).

Usage:
    python train_triplet_sam.py --epochs 30 --batch_size 8 --lr 1e-4
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL PERFORMANCE & DEVICE SETUP
# =============================================================================

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f" Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(" Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print(" Using CPU")


# =============================================================================
# SAM2 SETUP
# =============================================================================

SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "sam2/checkpoints/sam2.1_hiera_small.pt",
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_sam2_predictor = None


def get_sam2_predictor():
    """Lazy-load SAM2 predictor (or a dummy flag if unavailable)."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print(f"Loading SAM2 from {SAM2_CHECKPOINT}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 loaded!")
    except Exception as e:
        print(f"Warning: Could not load SAM2, using dummy masks. Error: {e}")
        _sam2_predictor = "dummy"

    return _sam2_predictor


def generate_water_mask(image_np, predictor):
    """Generate a binary water mask using SAM2 (or a dummy 1-mask)."""
    if predictor == "dummy":
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    try:
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, int(h * 0.7)]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = np.argmax(scores)
        return masks[best].astype(np.float32)
    except Exception as e:
        print(f"SAM2 mask generation failed, using dummy mask. Error: {e}")
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)


# =============================================================================
# MODEL
# =============================================================================


class TripletSAMModel(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_tiny_patch16_224",
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Create 4-channel backbone (RGB + mask)
        base_model = timm.create_model(backbone, pretrained=True, num_classes=0)

        if hasattr(base_model, "patch_embed"):
            old_proj = base_model.patch_embed.proj
            new_proj = nn.Conv2d(
                4,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
            )
            with torch.no_grad():
                # copy RGB weights
                new_proj.weight[:, :3] = old_proj.weight
                # init mask channel like first RGB channel
                new_proj.weight[:, 3:] = old_proj.weight[:, :1]
                new_proj.bias = old_proj.bias
            base_model.patch_embed.proj = new_proj

        self.backbone = base_model

        # Infer backbone feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 4, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]

        # Cross-attention between query & (ref1, ref2)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Elevation embedding (elev1, elev2)
        self.elev_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Head: [feat_q, attn_out, elev_emb]
        self.head = nn.Sequential(
            nn.Linear(backbone_dim * 2 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ref1, ref2, query, elev1, elev2):
        # [B, C, H, W] -> [B, F]
        feat1 = self.backbone(ref1)
        feat2 = self.backbone(ref2)
        feat_q = self.backbone(query)

        # refs: [B, 2, F]
        refs = torch.stack([feat1, feat2], dim=1)

        # Query attends to refs
        attn_out, _ = self.cross_attn(
            feat_q.unsqueeze(1),  # [B, 1, F]
            refs,                 # [B, 2, F]
            refs,
        )  # -> [B, 1, F]

        elev_pair = torch.stack([elev1, elev2], dim=1)  # [B, 2]
        elev_emb = self.elev_embed(elev_pair)           # [B, 64]

        fused = torch.cat(
            [feat_q, attn_out.squeeze(1), elev_emb],
            dim=1,
        )  # [B, 2F + 64]

        out = self.head(fused).squeeze(-1)  # [B]
        return out


# =============================================================================
# DATASET
# =============================================================================


class SameSiteTripletDatasetSAM(torch.utils.data.Dataset):
    """
    Triplet dataset (same site) with SAM2 water masks.

    Each item:
        ref1, ref2, query: [4, 224, 224] (RGB + mask)
        elevation1, elevation2, elevation_query: normalized floats
    """

    def __init__(
        self,
        csv_path,
        elev_mean=None,
        elev_std=None,
        min_elev_diff: float = 0.3,
        max_triplets: int = 5000,
        cache_masks: bool = True,
        mask_cache_dir: str = "data/mask_cache",
    ):
        self.df = pd.read_csv(csv_path)
        self.df["gage_height_ft"] = pd.to_numeric(
            self.df["gage_height_ft"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
        self.df = self.df[
            (self.df["gage_height_ft"] > 0)
            & (self.df["gage_height_ft"] < 50)
        ].reset_index(drop=True)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.5],
        )

        # Elevation normalization
        self.elev_mean = (
            elev_mean if elev_mean is not None else self.df["gage_height_ft"].mean()
        )
        self.elev_std = (
            elev_std if elev_std is not None else self.df["gage_height_ft"].std()
        )
        print(f"Normalization: mean={self.elev_mean:.2f}, std={self.elev_std:.2f}")

        self.cache_masks = cache_masks
        self.mask_cache_dir = Path(mask_cache_dir)
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

        self.sam_predictor = get_sam2_predictor()

        # Group by camera_id (sites with >=3 images)
        self.site_groups = {
            site: self.df[self.df["camera_id"] == site].reset_index(drop=True)
            for site in self.df["camera_id"].unique()
            if len(self.df[self.df["camera_id"] == site]) >= 3
        }
        print(f"Sites with 3+ images: {len(self.site_groups)}")

        # Build triplets
        self.triplets = []
        sites = list(self.site_groups.keys())
        np.random.seed(42)
        attempts = 0
        while len(self.triplets) < max_triplets and attempts < max_triplets * 20:
            site = np.random.choice(sites)
            site_df = self.site_groups[site]
            n = len(site_df)
            if n >= 3:
                i, j, k = np.random.choice(n, 3, replace=False)
                elev_i = site_df.iloc[i]["gage_height_ft"]
                elev_j = site_df.iloc[j]["gage_height_ft"]
                if abs(elev_i - elev_j) >= min_elev_diff:
                    self.triplets.append((site, i, j, k))
            attempts += 1
        print(f"Created {len(self.triplets)} triplets")

    def _load_image_with_mask(self, image_path: str) -> torch.Tensor:
        """Load RGB image + SAM2 water mask, return normalized 4-channel tensor."""
        img = Image.open(image_path).convert("RGB")
        img_resized = self.resize(img)
        img_np = np.array(img_resized)

        # Cache path: per site + image
        cache_path = (
            self.mask_cache_dir
            / Path(image_path).parent.parent.name
            / (Path(image_path).stem + "_mask.npy")
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_masks and cache_path.exists():
            mask = np.load(cache_path)
        else:
            mask = generate_water_mask(img_np, self.sam_predictor)
            # Resize mask to 224x224 and normalize to [0,1]
            mask = (
                np.array(
                    Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))
                )
                / 255.0
            ).astype(np.float32)
            if self.cache_masks:
                np.save(cache_path, mask)

        img_tensor = self.to_tensor(img_resized)     # [3, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        x4 = torch.cat([img_tensor, mask_tensor], dim=0)   # [4, H, W]
        return self.normalize(x4)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        site, i, j, k = self.triplets[idx]
        site_df = self.site_groups[site]

        row1, row2, row3 = site_df.iloc[i], site_df.iloc[j], site_df.iloc[k]

        e1 = (row1["gage_height_ft"] - self.elev_mean) / self.elev_std
        e2 = (row2["gage_height_ft"] - self.elev_mean) / self.elev_std
        eq = (row3["gage_height_ft"] - self.elev_mean) / self.elev_std

        return {
            "ref1": self._load_image_with_mask(row1["image_path"]),
            "ref2": self._load_image_with_mask(row2["image_path"]),
            "query": self._load_image_with_mask(row3["image_path"]),
            "elevation1": torch.tensor(e1, dtype=torch.float32),
            "elevation2": torch.tensor(e2, dtype=torch.float32),
            "elevation_query": torch.tensor(eq, dtype=torch.float32),
        }


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if len(preds) > 1:
        pearson_r = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson_r = 0.0
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": pearson_r,
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

MEAN_4 = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
STD_4 = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)


def denorm_4ch(img_4ch: torch.Tensor):
    """
    Inverse normalization for a 4-channel tensor.
    Returns: rgb_np (H,W,3), mask_np (H,W)
    """
    x = img_4ch.clone().cpu() * STD_4 + MEAN_4
    x = x.clamp(0.0, 1.0)
    rgb = x[:3].permute(1, 2, 0).numpy()
    mask = x[3].numpy()
    return rgb, mask


def plot_training_curves(train_losses, val_maes, output_path):
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, train_losses, marker="o")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE Loss")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, val_maes, marker="o", color="orange")
    ax[1].set_title("Validation MAE (ft)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("MAE (ft)")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter_preds_vs_targets(targets_ft, preds_ft, metrics, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets_ft, preds_ft, alpha=0.5, s=20, c="#FF5722")
    mn = min(targets_ft.min(), preds_ft.min()) - 0.5
    mx = max(targets_ft.max(), preds_ft.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], "r--", lw=2)
    ax.set_xlabel("Actual (ft)")
    ax.set_ylabel("Predicted (ft)")
    ax.set_title(
        f"Triplet+SAM: MAE={metrics['mae']:.3f} ft, R²={metrics['r2']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_sample_visualizations(
    samples,
    elev_mean,
    elev_std,
    output_dir,
    max_samples=4,
):
    """
    samples: list of dicts with keys:
        ref1, ref2, query: [4,H,W] tensors
        pred, target: normalized (float)
    """
    max_samples = min(max_samples, len(samples))
    for idx in range(max_samples):
        s = samples[idx]
        ref1_rgb, ref1_mask = denorm_4ch(s["ref1"])
        ref2_rgb, ref2_mask = denorm_4ch(s["ref2"])
        q_rgb, q_mask = denorm_4ch(s["query"])

        # Denormalize elevations back to ft
        target_ft = s["target"] * elev_std + elev_mean
        pred_ft = s["pred"] * elev_std + elev_mean

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))

        # Top row: RGB
        axes[0, 0].imshow(ref1_rgb)
        axes[0, 0].set_title("Ref1 RGB")
        axes[0, 1].imshow(ref2_rgb)
        axes[0, 1].set_title("Ref2 RGB")
        axes[0, 2].imshow(q_rgb)
        axes[0, 2].set_title("Query RGB")

        # Bottom row: SAM masks
        axes[1, 0].imshow(ref1_mask, cmap="Blues")
        axes[1, 0].set_title("Ref1 SAM Mask")
        axes[1, 1].imshow(ref2_mask, cmap="Blues")
        axes[1, 1].set_title("Ref2 SAM Mask")
        im = axes[1, 2].imshow(q_mask, cmap="Blues")
        axes[1, 2].set_title(
            f"Query SAM Mask\nTrue: {target_ft:.2f} ft | Pred: {pred_ft:.2f} ft"
        )

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(output_dir / f"sample_{idx+1}.png", dpi=150)
        plt.close(fig)


# =============================================================================
# TRAINING
# =============================================================================


def train(config):
    print("\n" + "=" * 70)
    print("🌊 TRIPLET + SAM MODEL TRAINING")
    print("=" * 70)

    # --------------------- datasets ---------------------
    train_dataset = SameSiteTripletDatasetSAM(
        config["train_csv"], max_triplets=config["max_triplets"]
    )
    val_dataset = SameSiteTripletDatasetSAM(
        config["val_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
    )
    test_dataset = SameSiteTripletDatasetSAM(
        config["test_csv"],
        elev_mean=train_dataset.elev_mean,
        elev_std=train_dataset.elev_std,
        max_triplets=1500,
    )

    # --------------------- dataloaders ------------------
    # If SAM2 is actually loaded, keep num_workers small to avoid GPU weirdness.
    if train_dataset.sam_predictor == "dummy":
        num_workers = 2
    else:
        num_workers = 0

    pin_memory = DEVICE.type == "cuda"
    persistent_workers = num_workers > 0

    batch_size = max(1, config["batch_size"] // 2)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **loader_kwargs
    )

    # --------------------- model & optimizer ------------
    model = TripletSAMModel(config["backbone"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # --------------------- output dir -------------------
    output_dir = Path("outputs") / f"triplet_sam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- training loop ----------------
    best_val_mae = float("inf")
    train_losses = []
    val_maes = []

    non_blocking = DEVICE.type == "cuda"

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
        )

        for batch in pbar:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
            query = batch["query"].to(DEVICE, non_blocking=non_blocking)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)
            targets = batch["elevation_query"].to(DEVICE, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)
                loss = F.mse_loss(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

            pbar.set_postfix({"loss": f"{train_loss_sum / n_train:.4f}"})

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        epoch_train_loss = train_loss_sum / n_train
        train_losses.append(epoch_train_loss)

        # ----------------- validation --------------------
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
                ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
                query = batch["query"].to(DEVICE, non_blocking=non_blocking)
                elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
                elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)

                with autocast(enabled=(DEVICE.type == "cuda")):
                    preds = model(ref1, ref2, query, elev1, elev2)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch["elevation_query"].numpy())

        # Denorm to ft
        val_preds_ft = np.array(val_preds) * train_dataset.elev_std + train_dataset.elev_mean
        val_targets_ft = np.array(val_targets) * train_dataset.elev_std + train_dataset.elev_mean
        val_metrics = compute_metrics(val_preds_ft, val_targets_ft)
        val_mae = val_metrics["mae"]
        val_maes.append(val_mae)

        scheduler.step()

        print(
            f"   Train Loss: {epoch_train_loss:.4f} | "
            f"Val MAE: {val_mae:.3f} ft | R²: {val_metrics['r2']:.3f}"
        )

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "elev_mean": train_dataset.elev_mean,
                    "elev_std": train_dataset.elev_std,
                },
                output_dir / "best_model.pt",
            )
            print("   ✓ Saved best model!")

    # ----------------- save training curves -------------
    plot_training_curves(
        train_losses,
        val_maes,
        output_dir / "training_curves.png",
    )

    # --------------------- test best model --------------
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    sample_vis_buffer = []  # store a few examples for visualization

    with torch.no_grad():
        for batch in test_loader:
            ref1 = batch["ref1"].to(DEVICE, non_blocking=non_blocking)
            ref2 = batch["ref2"].to(DEVICE, non_blocking=non_blocking)
            query = batch["query"].to(DEVICE, non_blocking=non_blocking)
            elev1 = batch["elevation1"].to(DEVICE, non_blocking=non_blocking)
            elev2 = batch["elevation2"].to(DEVICE, non_blocking=non_blocking)
            eq = batch["elevation_query"].to(DEVICE, non_blocking=non_blocking)

            with autocast(enabled=(DEVICE.type == "cuda")):
                preds = model(ref1, ref2, query, elev1, elev2)

            preds_np = preds.cpu().numpy()
            eq_np = eq.cpu().numpy()

            test_preds.extend(preds_np)
            test_targets.extend(eq_np)

            # store a few samples for visualization (on CPU)
            if len(sample_vis_buffer) < 6:  # cap number of examples
                for i in range(min(ref1.size(0), 6 - len(sample_vis_buffer))):
                    sample_vis_buffer.append(
                        {
                            "ref1": batch["ref1"][i].cpu(),
                            "ref2": batch["ref2"][i].cpu(),
                            "query": batch["query"][i].cpu(),
                            "pred": preds_np[i],
                            "target": eq_np[i],
                        }
                    )

    test_preds_ft = np.array(test_preds) * train_dataset.elev_std + train_dataset.elev_mean
    test_targets_ft = np.array(test_targets) * train_dataset.elev_std + train_dataset.elev_mean
    test_metrics = compute_metrics(test_preds_ft, test_targets_ft)

    # Scatter plot: Pred vs Actual
    plot_scatter_preds_vs_targets(
        test_targets_ft,
        test_preds_ft,
        test_metrics,
        output_dir / "test_predictions_scatter.png",
    )

    # Visualize a few samples (RGB + SAM + text with true vs pred)
    save_sample_visualizations(
        sample_vis_buffer,
        train_dataset.elev_mean,
        train_dataset.elev_std,
        output_dir,
        max_samples=4,
    )

    # Save JSON summary
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "model": "triplet_sam",
                "test_mae_ft": test_metrics["mae"],
                "test_mae_inches": test_metrics["mae"] * 12.0,
                "test_r2": test_metrics["r2"],
                "test_rmse_ft": test_metrics["rmse"],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(
        f" TRIPLET+SAM: MAE={test_metrics['mae']:.3f} ft "
        f"({test_metrics['mae'] * 12:.1f} in), R²={test_metrics['r2']:.3f}"
    )
    print(f"Output folder: {output_dir}")
    print("Saved:")
    print("  - training_curves.png")
    print("  - test_predictions_scatter.png")
    print("  - sample_1..N.png (RGB + SAM + prediction)")
    print("  - results.json")
    print("=" * 70)

    return test_metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(
        {
            "train_csv": "data/train_quality.csv",
            "val_csv": "data/val_quality.csv",
            "test_csv": "data/test_quality.csv",
            "backbone": "vit_tiny_patch16_224",
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "learning_rate": args.lr,
            "max_triplets": 5000,
        }
    )
