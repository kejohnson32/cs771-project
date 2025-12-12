#!/usr/bin/env python3
"""
Seen vs Unseen Evaluation Script

This script assumes you have already run:

    python scripts/download_eval_data.py

which creates:

    data/seen_eval.csv   -> SEEN   = validation or new images from training sites
    data/unseen_eval.csv -> UNSEEN = images from unseen sites (usually copy of test_quality.csv)

It then:

  * Loads each of the four trained models:
        - Siamese RGB      (models/siamese_rgb/best_model.pt)
        - Siamese SAM      (models/siamese_sam/best_model.pt)
        - Triplet RGB      (models/triplet_rgb/best_model.pt)
        - Triplet SAM      (models/triplet_sam/best_model.pt)
  * Builds SEEN and UNSEEN evaluation datasets for each model
  * Computes metrics: MAE, RMSE, R^2, Pearson r
  * Saves scatter plots for SEEN and UNSEEN
  * Writes a summary JSON to outputs/seen_unseen_results.json

Run from the project root:

    cd cs771-project
    source .venv/bin/activate
    python scripts/eval_seen_unseen.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------------
# PATHS & PYTHONPATH SETUP
# ------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
SCRIPT_DIR = THIS_FILE.parent

# make sure we can import train_*.py modules
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEEN_EVAL_CSV = DATA_DIR / "seen_eval.csv"
UNSEEN_EVAL_CSV = DATA_DIR / "unseen_eval.csv"

TRAIN_CSV = DATA_DIR / "train_quality.csv"
VAL_CSV = DATA_DIR / "val_quality.csv"
TEST_CSV = DATA_DIR / "test_quality.csv"

# ------------------------------------------------------------------------
# IMPORT MODEL / DATASET CLASSES FROM TRAINING SCRIPTS
# ------------------------------------------------------------------------

from train_siamese_rgb import (
    SiameseWaterLevelModel,
    SameSitePairDataset as RGBPairDataset,
)  # type: ignore
from train_triplet_rgb import (
    TripletWaterLevelModel,
    SameSiteTripletDataset as RGBTripletDataset,
)  # type: ignore
from train_siamese_sam import SiameseSAMModel, SameSitePairDatasetSAM  # type: ignore
from train_triplet_sam import TripletSAMModel, SameSiteTripletDatasetSAM  # type: ignore

# ------------------------------------------------------------------------
# DEVICE SELECTION
# ------------------------------------------------------------------------

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ------------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------------

def compute_metrics(preds_ft: np.ndarray, targets_ft: np.ndarray):
    preds_ft = np.asarray(preds_ft, dtype=float)
    targets_ft = np.asarray(targets_ft, dtype=float)

    mae = float(np.mean(np.abs(preds_ft - targets_ft)))
    rmse = float(np.sqrt(np.mean((preds_ft - targets_ft) ** 2)))

    ss_res = float(np.sum((targets_ft - preds_ft) ** 2))
    ss_tot = float(np.sum((targets_ft - np.mean(targets_ft)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if len(preds_ft) > 1:
        corr = float(np.corrcoef(preds_ft, targets_ft)[0, 1])
    else:
        corr = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": corr,
    }

# ------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------

def plot_scatter(preds_ft, targets_ft, metrics, title, outfile: Path):
    preds_ft = np.asarray(preds_ft, dtype=float)
    targets_ft = np.asarray(targets_ft, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets_ft, preds_ft, alpha=0.5, s=20)

    mn = min(targets_ft.min(), preds_ft.min())
    mx = max(targets_ft.max(), preds_ft.max())
    pad = 0.5
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "r--", linewidth=2)

    ax.set_xlabel("Actual Water Level (ft)")
    ax.set_ylabel("Predicted Water Level (ft)")
    ax.set_title(
        f"{title}\n"
        f"MAE: {metrics['mae']:.3f} ft ({metrics['mae']*12:.1f} in), "
        f"R²: {metrics['r2']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)

    plt.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------------
# EVALUATION HELPERS
# ------------------------------------------------------------------------

def evaluate_siamese_rgb(ckpt_path: Path, seen_csv: Path, unseen_csv: Path):
    print("\n" + "=" * 70)
    print("EVALUATING SIAMESE RGB MODEL")
    print("=" * 70)

    # Explicitly allow full checkpoint (PyTorch >= 2.6)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = SiameseWaterLevelModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    elev_mean = checkpoint["elev_mean"]
    elev_std = checkpoint["elev_std"]

    results = {}

    for split_name, csv_path in [("seen", seen_csv), ("unseen", unseen_csv)]:
        print(f"\n[{split_name.upper()}] using {csv_path}")
        ds = RGBPairDataset(
            str(csv_path),
            elev_mean=elev_mean,
            elev_std=elev_std,
            max_pairs=5000,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        preds_norm, targets_norm = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name}"):
                ref = batch["ref"].to(DEVICE)
                query = batch["query"].to(DEVICE)
                ref_elev = batch["ref_elevation"].to(DEVICE)
                target = batch["query_elevation"].to(DEVICE)

                pred = model(ref, query, ref_elev)

                preds_norm.append(pred.cpu().numpy())
                targets_norm.append(target.cpu().numpy())

        preds_norm = np.concatenate(preds_norm, axis=0)
        targets_norm = np.concatenate(targets_norm, axis=0)

        preds_ft = preds_norm * elev_std + elev_mean
        targets_ft = targets_norm * elev_std + elev_mean

        metrics = compute_metrics(preds_ft, targets_ft)
        results[split_name] = metrics

        plot_scatter(
            preds_ft,
            targets_ft,
            metrics,
            title=f"Siamese RGB ({split_name.upper()})",
            outfile=OUTPUTS_DIR / f"seen_unseen_siamese_rgb_{split_name}.png",
        )

        print(
            f"  {split_name.capitalize()} MAE:  {metrics['mae']:.3f} ft "
            f"({metrics['mae']*12:.1f} in), R²: {metrics['r2']:.3f}"
        )

    return results


def evaluate_triplet_rgb(ckpt_path: Path, seen_csv: Path, unseen_csv: Path):
    print("\n" + "=" * 70)
    print("EVALUATING TRIPLET RGB MODEL")
    print("=" * 70)

    # Load raw checkpoint (allow full object graph)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    raw_state = checkpoint["model_state_dict"]

    # ------------------------------------------------------------------
    # 1) Remap old key names -> new TripletWaterLevelModel names
    # ------------------------------------------------------------------
    remapped_state = {}

    for k, v in raw_state.items():
        new_k = k

        # backbone.model.*  -> backbone.*
        if new_k.startswith("backbone.model."):
            new_k = "backbone." + new_k[len("backbone.model.") :]

        # elevation_embed.*  -> elev_embed.*
        elif new_k.startswith("elevation_embed.0."):
            # first linear layer
            new_k = "elev_embed.0." + new_k[len("elevation_embed.0.") :]
        elif new_k.startswith("elevation_embed.1."):
            # second linear layer -> index 2 in our Sequential (index 1 is ReLU)
            new_k = "elev_embed.2." + new_k[len("elevation_embed.1.") :]

        # cross_attention.* -> cross_attn.*
        elif new_k.startswith("cross_attention."):
            new_k = "cross_attn." + new_k[len("cross_attention.") :]

        # head.head.* + head.output.* -> head.* in our Sequential:
        # head = [0:Linear, 1:ReLU, 2:Dropout, 3:Linear, 4:ReLU, 5:Linear]
        elif new_k.startswith("head.head.0."):
            # first linear in head
            new_k = "head.0." + new_k[len("head.head.0.") :]
        elif new_k.startswith("head.head.4."):
            # second linear in head (index 3)
            new_k = "head.3." + new_k[len("head.head.4.") :]
        elif new_k.startswith("head.output."):
            # final linear (index 5)
            new_k = "head.5." + new_k[len("head.output.") :]

        # We *don't* map keys like:
        #   self_attention.*, norm1/norm2, mlp.*, feature_proj.*, etc.
        # because those layers don't exist in the new architecture.
        # They will be dropped by simply not adding them.

        remapped_state[new_k] = v

    # ------------------------------------------------------------------
    # 2) Build model and only load parameters whose shapes match
    # ------------------------------------------------------------------
    model = TripletWaterLevelModel().to(DEVICE)
    model_state = model.state_dict()

    filtered_state = {}
    skipped_shape = []

    for k, v in remapped_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        # if k not in model_state: it's from old architecture; ignore

    load_result = model.load_state_dict(filtered_state, strict=False)

    print("  Triplet RGB remap summary:")
    print(f"    Loaded params:          {len(filtered_state)}")
    print(f"    Skipped (shape misfit): {len(skipped_shape)}")
    print(f"    Missing keys:           {len(load_result.missing_keys)}")
    print(f"    Unexpected keys:        {len(load_result.unexpected_keys)}")

    model.eval()

    elev_mean = checkpoint["elev_mean"]
    elev_std = checkpoint["elev_std"]

    results = {}

    for split_name, csv_path in [("seen", seen_csv), ("unseen", unseen_csv)]:
        print(f"\n[{split_name.upper()}] using {csv_path}")
        ds = RGBTripletDataset(
            str(csv_path),
            elev_mean=elev_mean,
            elev_std=elev_std,
            max_triplets=5000,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        preds_norm, targets_norm = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name}"):
                ref1 = batch["ref1"].to(DEVICE)
                ref2 = batch["ref2"].to(DEVICE)
                query = batch["query"].to(DEVICE)
                elev1 = batch["elevation1"].to(DEVICE)
                elev2 = batch["elevation2"].to(DEVICE)
                target = batch["elevation_query"].to(DEVICE)

                pred = model(ref1, ref2, query, elev1, elev2)

                preds_norm.append(pred.cpu().numpy())
                targets_norm.append(target.cpu().numpy())

        if not preds_norm:
            print(f"  No samples generated for {split_name}, skipping.")
            continue

        preds_norm = np.concatenate(preds_norm, axis=0)
        targets_norm = np.concatenate(targets_norm, axis=0)

        preds_ft = preds_norm * elev_std + elev_mean
        targets_ft = targets_norm * elev_std + elev_mean

        metrics = compute_metrics(preds_ft, targets_ft)
        results[split_name] = metrics

        plot_scatter(
            preds_ft,
            targets_ft,
            metrics,
            title=f"Triplet RGB ({split_name.upper()})",
            outfile=OUTPUTS_DIR / f"seen_unseen_triplet_rgb_{split_name}.png",
        )

        print(
            f"  {split_name.capitalize()} MAE:  {metrics['mae']:.3f} ft "
            f"({metrics['mae']*12:.1f} in), R²: {metrics['r2']:.3f}"
        )

    return results


def evaluate_siamese_sam(ckpt_path: Path, seen_csv: Path, unseen_csv: Path):
    print("\n" + "=" * 70)
    print("EVALUATING SIAMESE SAM MODEL")
    print("=" * 70)
    print("NOTE: This model uses SAM2. Make sure SAM2 dependencies are installed.")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = SiameseSAMModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    elev_mean = checkpoint["elev_mean"]
    elev_std = checkpoint["elev_std"]

    results = {}

    for split_name, csv_path in [("seen", seen_csv), ("unseen", unseen_csv)]:
        print(f"\n[{split_name.upper()}] using {csv_path}")
        ds = SameSitePairDatasetSAM(
            str(csv_path),
            elev_mean=elev_mean,
            elev_std=elev_std,
            max_pairs=2000,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        preds_norm, targets_norm = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name}"):
                ref = batch["ref"].to(DEVICE)
                query = batch["query"].to(DEVICE)
                ref_elev = batch["ref_elevation"].to(DEVICE)
                target = batch["query_elevation"].to(DEVICE)

                pred = model(ref, query, ref_elev)

                preds_norm.append(pred.cpu().numpy())
                targets_norm.append(target.cpu().numpy())

        if not preds_norm:
            print(f"  No samples generated for {split_name}, skipping.")
            continue

        preds_norm = np.concatenate(preds_norm, axis=0)
        targets_norm = np.concatenate(targets_norm, axis=0)

        preds_ft = preds_norm * elev_std + elev_mean
        targets_ft = targets_norm * elev_std + elev_mean

        metrics = compute_metrics(preds_ft, targets_ft)
        results[split_name] = metrics

        plot_scatter(
            preds_ft,
            targets_ft,
            metrics,
            title=f"Siamese SAM ({split_name.upper()})",
            outfile=OUTPUTS_DIR / f"seen_unseen_siamese_sam_{split_name}.png",
        )

        print(
            f"  {split_name.capitalize()} MAE:  {metrics['mae']:.3f} ft "
            f"({metrics['mae']*12:.1f} in), R²: {metrics['r2']:.3f}"
        )

    return results


def evaluate_triplet_sam(ckpt_path: Path, seen_csv: Path, unseen_csv: Path):
    print("\n" + "=" * 70)
    print("EVALUATING TRIPLET SAM MODEL")
    print("=" * 70)
    print("NOTE: This model uses SAM2. Make sure SAM2 dependencies are installed.")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = TripletSAMModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    elev_mean = checkpoint["elev_mean"]
    elev_std = checkpoint["elev_std"]

    results = {}

    for split_name, csv_path in [("seen", seen_csv), ("unseen", unseen_csv)]:
        print(f"\n[{split_name.upper()}] using {csv_path}")
        ds = SameSiteTripletDatasetSAM(
            str(csv_path),
            elev_mean=elev_mean,
            elev_std=elev_std,
            max_triplets=2000,
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        preds_norm, targets_norm = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name}"):
                ref1 = batch["ref1"].to(DEVICE)
                ref2 = batch["ref2"].to(DEVICE)
                query = batch["query"].to(DEVICE)
                elev1 = batch["elevation1"].to(DEVICE)
                elev2 = batch["elevation2"].to(DEVICE)
                target = batch["elevation_query"].to(DEVICE)

                pred = model(ref1, ref2, query, elev1, elev2)

                preds_norm.append(pred.cpu().numpy())
                targets_norm.append(target.cpu().numpy())

        if not preds_norm:
            print(f"  No samples generated for {split_name}, skipping.")
            continue

        preds_norm = np.concatenate(preds_norm, axis=0)
        targets_norm = np.concatenate(targets_norm, axis=0)

        preds_ft = preds_norm * elev_std + elev_mean
        targets_ft = targets_norm * elev_std + elev_mean

        metrics = compute_metrics(preds_ft, targets_ft)
        results[split_name] = metrics

        plot_scatter(
            preds_ft,
            targets_ft,
            metrics,
            title=f"Triplet SAM ({split_name.upper()})",
            outfile=OUTPUTS_DIR / f"seen_unseen_triplet_sam_{split_name}.png",
        )

        print(
            f"  {split_name.capitalize()} MAE:  {metrics['mae']:.3f} ft "
            f"({metrics['mae']*12:.1f} in), R²: {metrics['r2']:.3f}"
        )

    return results

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SEEN vs UNSEEN EVALUATION")
    print("=" * 70)
    print(f"Root directory: {ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")

    # --------------------------------------------------------------------
    # Check CSVs
    # --------------------------------------------------------------------
    if not SEEN_EVAL_CSV.exists():
        raise FileNotFoundError(
            f"{SEEN_EVAL_CSV} not found.\n"
            "Run `python scripts/download_eval_data.py` first to create it."
        )
    if not UNSEEN_EVAL_CSV.exists():
        raise FileNotFoundError(
            f"{UNSEEN_EVAL_CSV} not found.\n"
            "Run `python scripts/download_eval_data.py` first to create it."
        )

    print(f"\nUsing SEEN eval CSV:   {SEEN_EVAL_CSV}")
    print(f"Using UNSEEN eval CSV: {UNSEEN_EVAL_CSV}")

    seen_df = pd.read_csv(SEEN_EVAL_CSV)
    unseen_df = pd.read_csv(UNSEEN_EVAL_CSV)

    print(f"  SEEN:   {len(seen_df)} images from {seen_df['camera_id'].nunique()} sites")
    print(f"  UNSEEN: {len(unseen_df)} images from {unseen_df['camera_id'].nunique()} sites")

    results = {}

    # --------------------------------------------------------------------
    # Siamese RGB
    # --------------------------------------------------------------------
    siamese_rgb_ckpt = MODELS_DIR / "siamese_rgb" / "best_model.pt"
    if siamese_rgb_ckpt.exists():
        results["siamese_rgb"] = evaluate_siamese_rgb(
            siamese_rgb_ckpt, SEEN_EVAL_CSV, UNSEEN_EVAL_CSV
        )
    else:
        print("\n[WARNING] Siamese RGB checkpoint not found, skipping.")

    # --------------------------------------------------------------------
    # Triplet RGB
    # --------------------------------------------------------------------
    triplet_rgb_ckpt = MODELS_DIR / "triplet_rgb" / "best_model.pt"
    if triplet_rgb_ckpt.exists():
        results["triplet_rgb"] = evaluate_triplet_rgb(
            triplet_rgb_ckpt, SEEN_EVAL_CSV, UNSEEN_EVAL_CSV
        )
    else:
        print("\n[WARNING] Triplet RGB checkpoint not found, skipping.")

    # --------------------------------------------------------------------
    # Siamese SAM
    # --------------------------------------------------------------------
    siamese_sam_ckpt = MODELS_DIR / "siamese_sam" / "best_model.pt"
    if siamese_sam_ckpt.exists():
        results["siamese_sam"] = evaluate_siamese_sam(
            siamese_sam_ckpt, SEEN_EVAL_CSV, UNSEEN_EVAL_CSV
        )
    else:
        print("\n[WARNING] Siamese SAM checkpoint not found, skipping.")

    # --------------------------------------------------------------------
    # Triplet SAM
    # --------------------------------------------------------------------
    triplet_sam_ckpt = MODELS_DIR / "triplet_sam" / "best_model.pt"
    if triplet_sam_ckpt.exists():
        results["triplet_sam"] = evaluate_triplet_sam(
            triplet_sam_ckpt, SEEN_EVAL_CSV, UNSEEN_EVAL_CSV
        )
    else:
        print("\n[WARNING] Triplet SAM checkpoint not found, skipping.")

    # --------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY: SEEN vs UNSEEN")
        print("=" * 70)
        header = f"{'Model':<15} {'Seen MAE(ft)':<15} {'Seen R²':<10} {'Unseen MAE(ft)':<15} {'Unseen R²':<10}"
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            s = r.get("seen", {})
            u = r.get("unseen", {})
            print(
                f"{name:<15} "
                f"{s.get('mae', float('nan')):>10.3f}   "
                f"{s.get('r2', float('nan')):>8.3f}   "
                f"{u.get('mae', float('nan')):>10.3f}   "
                f"{u.get('r2', float('nan')):>8.3f}"
            )

        out_json = OUTPUTS_DIR / "seen_unseen_results.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_json}")
    else:
        print("\nNo models were evaluated (no checkpoints found).")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
