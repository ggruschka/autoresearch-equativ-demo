"""
CTR Prediction — Data Preparation & Evaluation Harness
=======================================================
DO NOT MODIFY THIS FILE. The agent modifies train.py only.

Downloads Criteo or Avazu sample data, preprocesses features, provides
dataloaders and evaluation utilities for train.py to consume.

Usage:
    uv run prepare.py                  # Download default (criteo)
    uv run prepare.py --dataset criteo
    uv run prepare.py --dataset avazu
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch-equativ-demo/ctr"))
TIME_BUDGET = 120  # seconds per experiment

# Criteo sample: first 1M rows of Criteo Display Ad Challenge (day_0)
# We generate synthetic data that mimics Criteo's distribution since the real
# dataset requires Kaggle authentication.
NUM_ROWS = 500_000
VAL_FRACTION = 0.2

# Feature spec for synthetic Criteo-like data
NUM_NUMERICAL = 13
NUM_CATEGORICAL = 26
CATEGORICAL_CARDINALITIES = [
    # Approximations of Criteo cardinalities, capped for embedding sanity
    1000, 500, 200_000, 50_000, 300, 20, 12_000, 600, 4, 50_000,
    8_000, 200_000, 30, 6_000, 100, 80_000, 50, 100, 40_000, 5,
    100_000, 20, 15, 200_000, 100, 50,
]
# Cap all cardinalities for hashing
HASH_BINS = [min(c, 10_000) for c in CATEGORICAL_CARDINALITIES]


@dataclass
class DatasetConfig:
    """Metadata about the loaded dataset."""
    name: str
    num_numerical: int
    num_categorical: int
    categorical_cardinalities: list[int]  # after hashing
    num_train: int
    num_val: int


# ---------------------------------------------------------------------------
# Synthetic data generation (deterministic, mimics Criteo distribution)
# ---------------------------------------------------------------------------

def _generate_criteo_like(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic CTR data mimicking Criteo distribution.

    Returns (numerical_features, categorical_features, labels).
    """
    rng = np.random.RandomState(seed)

    # Numerical features: mix of count-like and normalized features
    numerical = np.zeros((n, NUM_NUMERICAL), dtype=np.float32)
    for i in range(NUM_NUMERICAL):
        if i < 5:
            # Count features (log-normal)
            numerical[:, i] = rng.lognormal(mean=1.0, sigma=2.0, size=n).clip(0, 1000)
        else:
            # Normalized features
            numerical[:, i] = rng.randn(n).astype(np.float32)

    # Categorical features
    categorical = np.zeros((n, NUM_CATEGORICAL), dtype=np.int64)
    for i in range(NUM_CATEGORICAL):
        # Zipf-like distribution (some values much more common)
        bins = HASH_BINS[i]
        categorical[:, i] = (rng.zipf(1.5, size=n) % bins).astype(np.int64)

    # Labels: ~3.4% CTR (realistic for display ads)
    # Create a signal from features so the problem is learnable
    weights_num = rng.randn(NUM_NUMERICAL).astype(np.float32) * 0.1
    logits = numerical @ weights_num

    # Add categorical signal via simple hash trick
    for i in range(min(10, NUM_CATEGORICAL)):  # first 10 categoricals are informative
        cat_weights = rng.randn(HASH_BINS[i]).astype(np.float32) * 0.3
        logits += cat_weights[categorical[:, i]]

    # Shift to get ~3.4% CTR
    logits -= 3.3
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.rand(n) < probs).astype(np.float32)

    print(f"  CTR rate: {labels.mean():.4f} ({labels.sum():.0f}/{n})")
    return numerical, categorical, labels


def _generate_avazu_like(n: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic Avazu-like data (mobile-focused, different distribution)."""
    rng = np.random.RandomState(seed)

    # Avazu has fewer numerical features, more categorical
    num_feats = 6
    cat_feats = 20
    cat_bins = [24, 7, 500, 300, 5000, 8000, 50, 100, 3000, 200,
                400, 50, 20, 10, 100, 1000, 500, 200, 50, 10]

    numerical = rng.randn(n, num_feats).astype(np.float32)
    categorical = np.zeros((n, cat_feats), dtype=np.int64)
    for i in range(cat_feats):
        categorical[:, i] = (rng.zipf(1.3, size=n) % cat_bins[i]).astype(np.int64)

    # ~17% CTR (mobile ads have higher CTR)
    weights = rng.randn(num_feats).astype(np.float32) * 0.15
    logits = numerical @ weights
    for i in range(min(8, cat_feats)):
        cat_w = rng.randn(cat_bins[i]).astype(np.float32) * 0.25
        logits += cat_w[categorical[:, i]]
    logits -= 1.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.rand(n) < probs).astype(np.float32)

    print(f"  CTR rate: {labels.mean():.4f} ({labels.sum():.0f}/{n})")

    return numerical, categorical, labels


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _normalize_numerical(train_num: np.ndarray, val_num: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalization using train stats. Replace NaN with 0."""
    mean = np.nanmean(train_num, axis=0)
    std = np.nanstd(train_num, axis=0) + 1e-8
    train_out = (train_num - mean) / std
    val_out = (val_num - mean) / std
    train_out = np.nan_to_num(train_out, 0.0).astype(np.float32)
    val_out = np.nan_to_num(val_out, 0.0).astype(np.float32)
    return train_out, val_out


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(dataset: str) -> Path:
    return CACHE_DIR / dataset


def _save_arrays(path: Path, **arrays: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(path / f"{name}.npy", arr)


def _load_arrays(path: Path, *names: str) -> tuple[np.ndarray, ...]:
    return tuple(np.load(path / f"{name}.npy") for name in names)


def _is_cached(dataset: str) -> bool:
    p = _cache_path(dataset)
    return (p / "train_numerical.npy").exists() and (p / "config.npy").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(dataset: str = "criteo") -> DatasetConfig:
    """Download/generate and preprocess dataset. Returns config."""
    cache = _cache_path(dataset)

    if _is_cached(dataset):
        print(f"Dataset '{dataset}' already cached at {cache}")
        config_data = np.load(cache / "config.npy", allow_pickle=True).item()
        return DatasetConfig(**config_data)

    print(f"Generating synthetic '{dataset}' data...")

    if dataset == "criteo":
        numerical, categorical, labels = _generate_criteo_like(NUM_ROWS)
        cat_cardinalities = HASH_BINS
    elif dataset == "avazu":
        numerical, categorical, labels = _generate_avazu_like(NUM_ROWS)
        cat_cardinalities = [24, 7, 500, 300, 5000, 8000, 50, 100, 3000, 200,
                             400, 50, 20, 10, 100, 1000, 500, 200, 50, 10]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Chronological split (last VAL_FRACTION rows = validation)
    split_idx = int(len(labels) * (1 - VAL_FRACTION))
    train_num, val_num = numerical[:split_idx], numerical[split_idx:]
    train_cat, val_cat = categorical[:split_idx], categorical[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Normalize numerical features
    train_num, val_num = _normalize_numerical(train_num, val_num)

    print(f"  Train: {len(train_labels)} rows, Val: {len(val_labels)} rows")

    _save_arrays(
        cache,
        train_numerical=train_num, val_numerical=val_num,
        train_categorical=train_cat, val_categorical=val_cat,
        train_labels=train_labels, val_labels=val_labels,
    )

    config = DatasetConfig(
        name=dataset,
        num_numerical=train_num.shape[1],
        num_categorical=train_cat.shape[1],
        categorical_cardinalities=cat_cardinalities,
        num_train=len(train_labels),
        num_val=len(val_labels),
    )
    np.save(cache / "config.npy", config.__dict__)

    print(f"  Cached to {cache}")
    return config


def load_config(dataset: str = "criteo") -> DatasetConfig:
    """Load dataset config (must call prepare first)."""
    if not _is_cached(dataset):
        raise RuntimeError(f"Dataset '{dataset}' not prepared. Run: uv run prepare.py --dataset {dataset}")
    config_data = np.load(_cache_path(dataset) / "config.npy", allow_pickle=True).item()
    return DatasetConfig(**config_data)


def make_dataloader(dataset: str, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a PyTorch DataLoader for the given split.

    Returns batches of (numerical_tensor, categorical_tensor, label_tensor).
    """
    cache = _cache_path(dataset)
    num = torch.from_numpy(np.load(cache / f"{split}_numerical.npy"))
    cat = torch.from_numpy(np.load(cache / f"{split}_categorical.npy"))
    labels = torch.from_numpy(np.load(cache / f"{split}_labels.npy"))
    ds = TensorDataset(num, cat, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=(split == "train"))


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: str = "criteo",
             batch_size: int = 4096, device: str = "cpu") -> dict[str, float]:
    """Evaluate model on validation set. Returns dict with val_logloss and val_auc."""
    from sklearn.metrics import log_loss, roc_auc_score

    model.eval()
    loader = make_dataloader(dataset, "val", batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    for num, cat, labels in loader:
        num = num.to(device)
        cat = cat.to(device)
        logits = model(num, cat).squeeze(-1)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Clip predictions to avoid log(0)
    all_preds = np.clip(all_preds, 1e-7, 1 - 1e-7)

    val_logloss = log_loss(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, all_preds)

    model.train()
    return {"val_logloss": val_logloss, "val_auc": val_auc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CTR dataset")
    parser.add_argument("--dataset", default="criteo", choices=["criteo", "avazu"])
    args = parser.parse_args()

    config = prepare(args.dataset)
    print(f"\nDataset config:")
    print(f"  Name:              {config.name}")
    print(f"  Numerical feats:   {config.num_numerical}")
    print(f"  Categorical feats: {config.num_categorical}")
    print(f"  Cardinalities:     {config.categorical_cardinalities[:5]}... (showing first 5)")
    print(f"  Train rows:        {config.num_train}")
    print(f"  Val rows:          {config.num_val}")
    print(f"\nReady for training!")
