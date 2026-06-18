"""
Create a deterministic train/val/test split (70/15/15) from the dataset.

This script loads the dataset and creates a new split with a reproducible
random seed, saving results to split_dataset.csv.

Usage:
    python create_split.py [--seed SEED]
    
    Example:
        python create_split.py --seed 42
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd


def create_train_val_test_split(seed: int = 42) -> None:
    """
    Reseparate the dataset into train/val/test split (70/15/15) with deterministic seed.
    
    Loads the dataset, creates a new deterministic split, and saves the results to
    split_dataset.csv with columns: image_url, label_numeric, and data_split.
    
    Args:
        seed: Random seed for reproducibility (default: 42)
    """
    # Load the original dataset (FINAL_DATASET.csv specifically)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "datasets", "**", "FINAL_DATASET.csv")
    csv_files = glob.glob(dataset_path, recursive=True)
    if not csv_files:
        raise FileNotFoundError("FINAL_DATASET.csv not found — run downloaddataset.py first.")
    
    dataset_df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(dataset_df)} rows from {csv_files[0]}")
    
    # Extract only the columns we need
    data = dataset_df[["image_url", "label_numeric"]].copy()
    data = data.dropna()
    
    print(f"Using {len(data)} rows with valid image_url and label_numeric")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create indices and shuffle
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Reorder the rows themselves into the shuffled order, so the saved CSV is
    # shuffled — not just the split labels. The source dataset is grouped by
    # class; without physically reordering, every row keeps its original
    # position and the classes stay block-sorted in the output, so any consumer
    # that reads the rows sequentially (e.g. a streaming loader with a sample
    # cap) sees one class before the other.
    data = data.iloc[indices].reset_index(drop=True)

    # Calculate split points
    n_samples = len(data)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size

    # Assign splits to contiguous ranges of the now-shuffled rows. Because the
    # rows are already shuffled, contiguous slices are themselves random samples
    # and the per-row assignment matches the previous index-based logic.
    data['data_split'] = (
        ['train'] * train_size + ['val'] * val_size + ['test'] * test_size
    )

    # Display split statistics
    print(f"\nSplit Statistics:")
    print(f"  Train: {train_size} samples ({100 * train_size / n_samples:.1f}%)")
    print(f"  Val:   {val_size} samples ({100 * val_size / n_samples:.1f}%)")
    print(f"  Test:  {test_size} samples ({100 * test_size / n_samples:.1f}%)")
    
    # Save to datasets/split_dataset.csv
    datasets_dir = os.path.join(script_dir, "datasets")
    output_path = os.path.join(datasets_dir, "split_dataset.csv")
    data.to_csv(output_path, index=False)
    print(f"\nSplit dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test split (70/15/15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    create_train_val_test_split(seed=args.seed)
