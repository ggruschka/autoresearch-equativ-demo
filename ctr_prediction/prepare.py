"""
CTR Prediction — Data Preparation & Evaluation Harness
=======================================================
DO NOT MODIFY THIS FILE. The agent modifies train.py only.

Loads the real Criteo Display Advertising Challenge dataset, preprocesses
features, provides dataloaders and evaluation utilities for train.py to consume.

Usage:
    uv run prepare.py                  # Prepare default (criteo)
    uv run prepare.py --dataset criteo
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch-equativ-demo/ctr"))
RAW_DIR = CACHE_DIR / "criteo-raw"
TIME_BUDGET = 300  # seconds per experiment

# Use first 1M rows of the real Criteo dataset
NUM_ROWS = 1_000_000
VAL_FRACTION = 0.2

NUM_NUMERICAL = 13
NUM_CATEGORICAL = 26
# Hash all categorical features to at most this many bins
HASH_BINS_CAP = 10_000


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
# Real Criteo data loading
# ---------------------------------------------------------------------------

def _load_criteo_raw(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load first n rows of the real Criteo DAC train.txt.

    Format: <label>\t<int1>\t...\t<int13>\t<cat1>\t...\t<cat26>
    Missing values are empty strings.

    Returns (numerical, categorical, labels).
    """
    raw_path = RAW_DIR / "train.txt"
    if not raw_path.exists():
        raise RuntimeError(
            f"Criteo dataset not found at {raw_path}. "
            f"Please extract kaggle-display-advertising-challenge-dataset.tar.gz "
            f"into {RAW_DIR}/"
        )

    print(f"Loading {n:,} rows from {raw_path}...")

    labels = np.zeros(n, dtype=np.float32)
    numerical = np.zeros((n, NUM_NUMERICAL), dtype=np.float32)
    categorical = np.zeros((n, NUM_CATEGORICAL), dtype=np.int64)

    with open(raw_path, 'r') as f:
        for i in range(n):
            line = f.readline()
            if not line:
                print(f"  Warning: only {i} rows available")
                labels = labels[:i]
                numerical = numerical[:i]
                categorical = categorical[:i]
                break

            parts = line.rstrip('\n').split('\t')
            # Label
            labels[i] = float(parts[0])

            # 13 integer (numerical) features — columns 1-13
            for j in range(NUM_NUMERICAL):
                val = parts[1 + j]
                if val == '':
                    numerical[i, j] = np.nan
                else:
                    numerical[i, j] = float(val)

            # 26 categorical features — columns 14-39
            # Hash hex strings to integer indices
            for j in range(NUM_CATEGORICAL):
                val = parts[14 + j]
                if val == '':
                    categorical[i, j] = 0  # missing → index 0
                else:
                    categorical[i, j] = (int(val, 16) % (HASH_BINS_CAP - 1)) + 1

            if (i + 1) % 200_000 == 0:
                print(f"  Loaded {i + 1:,} rows...")

    print(f"  CTR rate: {labels.mean():.4f} ({labels.sum():.0f}/{len(labels)})")
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
    """Load and preprocess dataset. Returns config."""
    cache = _cache_path(dataset)

    if _is_cached(dataset):
        print(f"Dataset '{dataset}' already cached at {cache}")
        config_data = np.load(cache / "config.npy", allow_pickle=True).item()
        return DatasetConfig(**config_data)

    print(f"Preparing real '{dataset}' data...")

    if dataset == "criteo":
        numerical, categorical, labels = _load_criteo_raw(NUM_ROWS)
        cat_cardinalities = [HASH_BINS_CAP] * NUM_CATEGORICAL
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
    parser.add_argument("--dataset", default="criteo", choices=["criteo"])
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
