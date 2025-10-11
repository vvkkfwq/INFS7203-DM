"""
I/O utilities for saving and loading model results.

This module provides functions for:
- Saving training results to CSV
- Saving tuning results (summary and detailed)
- Loading previous results
- Creating output directories
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


def ensure_dir(directory: Path) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory (Path): Directory path

    Returns:
        Path: The directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_baseline_results(
    results: Dict[str, Any],
    output_file: str = "results/baseline_results.csv",
    append: bool = False,
) -> None:
    """
    Save baseline model results to CSV.

    Args:
        results (dict): Results dictionary containing:
            - model_name: Name of the model
            - accuracy_mean, accuracy_std: Accuracy statistics
            - f1_mean, f1_std: F1 statistics
            - parameters: Model parameters
        output_file (str): Output file path
        append (bool): Whether to append to existing file
    """
    output_path = Path(output_file)
    ensure_dir(output_path.parent)

    # Prepare summary data
    summary = {
        "model": results["model_name"],
        "accuracy_mean": results["cv_scores"]["accuracy_mean"],
        "accuracy_std": results["cv_scores"]["accuracy_std"],
        "f1_mean": results["cv_scores"]["f1_mean"],
        "f1_std": results["cv_scores"]["f1_std"],
    }

    # Convert to DataFrame
    df = pd.DataFrame([summary])

    # Save or append
    if append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def save_tuning_results(
    results: Dict[str, Any],
    output_file: str,
    save_detailed: bool = True,
) -> None:
    """
    Save hyperparameter tuning results to CSV.

    Args:
        results (dict): Tuning results dictionary containing:
            - model_name: Name of the model
            - best_f1_score, f1_std: F1 statistics
            - baseline_f1: Baseline F1 score
            - f1_improvement: Improvement over baseline
            - best_params: Best hyperparameters
            - cv_results (optional): Detailed CV results DataFrame
        output_file (str): Output file path
        save_detailed (bool): Whether to save detailed CV results
    """
    output_path = Path(output_file)
    ensure_dir(output_path.parent)

    # Prepare summary data
    summary = {
        "model": results["model_name"],
        "f1_mean": results["best_f1_score"],
        "f1_std": results["f1_std"],
        "baseline_f1": results.get("baseline_f1", None),
        "f1_improvement": results.get("f1_improvement", None),
    }

    # Add parameters to summary
    if "best_params" in results:
        summary.update({f"param_{k}": v for k, v in results["best_params"].items()})

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Tuning results saved to: {output_path}")

    # Save detailed CV results if available
    if save_detailed and "cv_results" in results:
        detailed_file = str(output_path).replace(".csv", "_detailed.csv")
        results["cv_results"].to_csv(detailed_file, index=False)
        print(f"✓ Detailed CV results saved to: {detailed_file}")


def save_model(
    model, preprocessor, output_file: str = "models/trained_model.pkl"
) -> None:
    """
    Save trained model and preprocessor to pickle file.

    Args:
        model: Trained sklearn model
        preprocessor: Fitted preprocessor
        output_file (str): Output file path
    """
    output_path = Path(output_file)
    ensure_dir(output_path.parent)

    # Save model and preprocessor together
    model_data = {"model": model, "preprocessor": preprocessor}

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to: {output_path}")


def load_model(model_file: str) -> Dict[str, Any]:
    """
    Load trained model and preprocessor from pickle file.

    Args:
        model_file (str): Model file path

    Returns:
        dict: Dictionary containing 'model' and 'preprocessor'
    """
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    print(f"\n✓ Model loaded from: {model_file}")
    return model_data


def load_baseline_results(results_file: str = "results/baseline_results.csv") -> pd.DataFrame:
    """
    Load baseline results from CSV.

    Args:
        results_file (str): Results file path

    Returns:
        pd.DataFrame: Baseline results table
    """
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_path)
    return df


def save_predictions(
    predictions, output_file: str, cv_scores: Optional[Dict[str, float]] = None
) -> None:
    """
    Save predictions to submission format file.

    Format:
        - Lines 1-N: Test predictions (0, or 1,)
        - Line N+1: CV results (accuracy,f1,)

    Args:
        predictions: Array of predictions
        output_file (str): Output file path (e.g., 's4860387.infs4203')
        cv_scores (dict, optional): CV scores with 'accuracy' and 'f1' keys
    """
    output_path = Path(output_file)
    ensure_dir(output_path.parent)

    with open(output_path, "w") as f:
        # Write predictions (one per line with comma)
        for pred in predictions:
            f.write(f"{pred},\n")

        # Write CV scores if provided
        if cv_scores:
            f.write(f"{cv_scores['accuracy']:.3f},{cv_scores['f1']:.3f},\n")

    print(f"\n✓ Predictions saved to: {output_path}")
