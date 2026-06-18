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
    
    # Calculate split points
    n_samples = len(data)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create split column
    data_split = [''] * len(data)
    for idx in train_indices:
        data_split[idx] = 'train'
    for idx in val_indices:
        data_split[idx] = 'val'
    for idx in test_indices:
        data_split[idx] = 'test'
    
    data['data_split'] = data_split
    
    # Display split statistics
    print(f"\nSplit Statistics:")
    print(f"  Train: {len(train_indices)} samples ({100 * len(train_indices) / n_samples:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({100 * len(val_indices) / n_samples:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({100 * len(test_indices) / n_samples:.1f}%)")
    
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
