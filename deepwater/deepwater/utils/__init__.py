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
