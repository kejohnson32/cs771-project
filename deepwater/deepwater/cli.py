"""
DeepWater CLI

Command-line interface for water level estimation pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def discover_sites(args):
    """Discover and rank camera sites."""
    from deepwater.data import discover_best_sites
    
    states = args.states.split(",") if args.states else None
    
    sites = discover_best_sites(
        num_sites=args.num_sites,
        states=states,
        output_path=args.output,
        min_images=args.min_images,
        min_elevation_range_ft=args.min_elevation_range,
    )
    
    print(f"\nDiscovered {len(sites)} sites")
    print(f"Saved to: {args.output}")
    
    # Print top sites
    print(f"\nTop {min(10, len(sites))} sites:")
    for i, site in enumerate(sites[:10]):
        print(f"  {i+1}. {site.camera_id}")
        print(f"     Site ID: {site.site_id}")
        print(f"     Quality Score: {site.quality_score:.1f}")
        if site.elevation_range_ft:
            print(f"     Elevation Range: {site.elevation_range_ft:.2f} ft")


def collect_data(args):
    """Collect training data from sites."""
    import json
    from deepwater.data import DataCollector, CollectionConfig
    
    # Load site list
    with open(args.sites, "r") as f:
        sites_data = json.load(f)
    
    # Convert to list of dicts
    sites = [
        {"camera_id": s["camera_id"], "site_id": s["site_id"]}
        for s in sites_data[:args.max_sites]
    ]
    
    config = CollectionConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        max_images_per_site=args.max_images,
    )
    
    with DataCollector(args.output, config) as collector:
        collector.collect_multiple_sites(sites, parallel=not args.sequential)
        collector.create_combined_dataset()
        
        summary = collector.get_collection_summary()
        print("\nCollection Summary:")
        print(summary.to_string())


def train_model(args):
    """Train a water level estimation model."""
    import torch
    from deepwater.data import WaterLevelDataset, split_dataset, create_dataloaders
    from deepwater.models import create_model, create_model_from_config
    from deepwater.training import WaterLevelTrainer, create_optimizer, create_scheduler
    
    # Split dataset if needed
    if args.split:
        train_csv, val_csv, test_csv = split_dataset(
            args.data,
            args.output_dir,
            stratify_by="00065" if args.stratify else None,
        )
    else:
        # Assume pre-split data
        data_dir = Path(args.data).parent
        train_csv = str(data_dir / "train.csv")
        val_csv = str(data_dir / "val.csv")
        test_csv = str(data_dir / "test.csv")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.model_type,
        image_size=args.image_size,
    )
    
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    
    # Create model
    if args.model_config:
        model = create_model_from_config(args.model_config)
    else:
        model = create_model(
            model_type=args.model_type,
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
        )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    
    # Create trainer
    trainer = WaterLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_amp=not args.no_amp,
        use_wandb=args.wandb,
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume,
    )
    
    # Evaluate on test set
    if test_loader is not None:
        from deepwater.utils import evaluate_model, print_metrics, plot_predictions
        
        predictions, targets, metrics = evaluate_model(
            model,
            test_loader,
            device=args.device,
            denormalize_fn=train_loader.dataset.denormalize_elevation,
        )
        
        print_metrics(metrics, title="Test Set Results")
        
        # Save plots
        plot_predictions(
            predictions, targets,
            title="Test Set Predictions",
            save_path=str(Path(args.output_dir) / trainer.experiment_name / "test_predictions.png"),
        )


def evaluate(args):
    """Evaluate a trained model."""
    import torch
    from deepwater.data import WaterLevelDataset
    from deepwater.models import create_model
    from deepwater.utils import evaluate_model, print_metrics, plot_predictions
    from torch.utils.data import DataLoader
    
    # Load model
    checkpoint = torch.load(args.model, map_location=args.device)
    
    # Recreate model (you'd need to save model config with checkpoint)
    model = create_model(
        model_type=args.model_type,
        backbone=args.backbone,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create test dataset
    test_dataset = WaterLevelDataset(
        data_csv=args.data,
        images_dir=args.images_dir,
        mode=args.model_type,
        augment=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Evaluate
    predictions, targets, metrics = evaluate_model(
        model,
        test_loader,
        device=args.device,
        denormalize_fn=test_dataset.denormalize_elevation,
    )
    
    print_metrics(metrics, title="Evaluation Results")
    
    if args.output:
        plot_predictions(
            predictions, targets,
            title="Predictions vs Targets",
            save_path=args.output,
        )


def main():
    parser = argparse.ArgumentParser(
        description="DeepWater - Water Level Estimation from Imagery",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Discover sites command
    discover_parser = subparsers.add_parser("discover", help="Discover camera sites")
    discover_parser.add_argument("--num-sites", type=int, default=50, help="Number of sites")
    discover_parser.add_argument("--states", type=str, help="Comma-separated state codes")
    discover_parser.add_argument("--output", type=str, default="data/site_rankings.json")
    discover_parser.add_argument("--min-images", type=int, default=100)
    discover_parser.add_argument("--min-elevation-range", type=float, default=1.0)
    discover_parser.set_defaults(func=discover_sites)
    
    # Collect data command
    collect_parser = subparsers.add_parser("collect", help="Collect training data")
    collect_parser.add_argument("--sites", type=str, required=True, help="Sites JSON file")
    collect_parser.add_argument("--output", type=str, default="data/collected")
    collect_parser.add_argument("--max-sites", type=int, default=50)
    collect_parser.add_argument("--max-images", type=int, default=2000)
    collect_parser.add_argument("--start-date", type=str)
    collect_parser.add_argument("--end-date", type=str)
    collect_parser.add_argument("--sequential", action="store_true")
    collect_parser.set_defaults(func=collect_data)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", type=str, required=True, help="Data CSV")
    train_parser.add_argument("--images-dir", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, default="outputs")
    train_parser.add_argument("--experiment-name", type=str)
    train_parser.add_argument("--model-type", type=str, default="siamese", 
                              choices=["siamese", "triplet"])
    train_parser.add_argument("--model-config", type=str, 
                              help="Predefined model config name")
    train_parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--optimizer", type=str, default="adamw")
    train_parser.add_argument("--scheduler", type=str, default="cosine")
    train_parser.add_argument("--warmup-epochs", type=int, default=5)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--split", action="store_true", help="Split dataset")
    train_parser.add_argument("--stratify", action="store_true")
    train_parser.set_defaults(func=train_model)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    eval_parser.add_argument("--data", type=str, required=True, help="Test data CSV")
    eval_parser.add_argument("--images-dir", type=str, required=True)
    eval_parser.add_argument("--model-type", type=str, default="siamese")
    eval_parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--output", type=str, help="Output plot path")
    eval_parser.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
