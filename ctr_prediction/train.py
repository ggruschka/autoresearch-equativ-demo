"""
CTR Prediction — Training Script
=================================
This file is modified by the AI agent. It contains the model architecture,
optimizer configuration, and training loop.

Run: uv run train.py
"""
from __future__ import annotations

import copy
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_config, make_dataloader, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# Hyperparameters (agent tunes these)
# ---------------------------------------------------------------------------

DATASET = "criteo"
EMBEDDING_DIM = 16
HIDDEN_DIMS = [128, 64]
LEARNING_RATE = 9e-4
DROPOUT = 0.2
BATCH_SIZE = 1024
WEIGHT_DECAY = 1e-4
NUM_BINS = 48  # bins for numerical feature embedding

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CTRModel(nn.Module):
    """MLP with learned embeddings for categorical features."""

    def __init__(self, config):
        super().__init__()

        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, EMBEDDING_DIM)
            for cardinality in config.categorical_cardinalities
        ])

        # Numerical feature binning: embed binned numerical features
        self.num_numerical = config.num_numerical
        # Bin boundaries: quantile-based for N(0,1) z-scored data
        # More bins in the dense center, fewer in tails
        quantiles = torch.linspace(0, 1, NUM_BINS + 1)[1:-1]  # exclude 0 and 1
        # Approximate N(0,1) quantile via probit: sqrt(2) * erfinv(2*q - 1)
        boundaries = math.sqrt(2) * torch.erfinv(2 * quantiles - 1)
        self.register_buffer('bin_boundaries', boundaries)
        # One embedding per numerical feature
        self.num_embeddings = nn.ModuleList([
            nn.Embedding(NUM_BINS, EMBEDDING_DIM)
            for _ in range(config.num_numerical)
        ])

        # Input dim = raw numerical + cat embeddings + numerical embeddings
        input_dim = config.num_numerical + config.num_categorical * EMBEDDING_DIM + config.num_numerical * EMBEDDING_DIM

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        # Embed each categorical feature
        cat_embedded = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]
        cat_embedded = torch.cat(cat_embedded, dim=-1)
        if self.training:
            cat_embedded = F.dropout(cat_embedded, p=0.1)

        # Piecewise linear embedding of numerical features
        bin_indices = torch.bucketize(numerical, self.bin_boundaries)  # (batch, 13)
        # Clamp to valid range for interpolation
        bin_lo = bin_indices.clamp(max=NUM_BINS - 2)
        bin_hi = bin_lo + 1
        # Compute interpolation weight within each bin
        all_bounds = torch.cat([
            torch.tensor([-1e6], device=numerical.device),
            self.bin_boundaries,
            torch.tensor([1e6], device=numerical.device),
        ])
        lo_val = all_bounds[bin_lo]  # (batch, 13)
        hi_val = all_bounds[bin_hi]  # (batch, 13)
        weight = ((numerical - lo_val) / (hi_val - lo_val + 1e-8)).clamp(0, 1)  # (batch, 13)

        num_embedded = []
        for i in range(self.num_numerical):
            emb_lo = self.num_embeddings[i](bin_lo[:, i])  # (batch, emb_dim)
            emb_hi = self.num_embeddings[i](bin_hi[:, i])  # (batch, emb_dim)
            w = weight[:, i:i+1]  # (batch, 1)
            num_embedded.append((1 - w) * emb_lo + w * emb_hi)
        num_embedded = torch.cat(num_embedded, dim=-1)

        # Concatenate: raw numerical + cat embeddings + num embeddings
        x = torch.cat([numerical, cat_embedded, num_embedded], dim=-1)

        return self.mlp(x)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load config
    config = load_config(DATASET)
    print(f"Dataset: {config.name} | Train: {config.num_train} | Val: {config.num_val}")

    # Build model
    model = CTRModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Data
    train_loader = make_dataloader(DATASET, "train", batch_size=BATCH_SIZE, shuffle=True)

    # LR schedule: linear warmup + cosine decay
    steps_per_epoch = config.num_train // BATCH_SIZE
    total_steps_est = steps_per_epoch * 9  # ~9 epochs in 300s
    warmup_steps = 500

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps_est - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA: accumulate averaged weights from last 25% of training
    swa_start_frac = 0.75
    swa_state = None
    swa_count = 0

    # Training
    t0 = time.time()
    step = 0
    epoch = 0
    best_logloss = float("inf")

    while True:
        epoch += 1
        for num, cat, labels in train_loader:
            if time.time() - t0 > TIME_BUDGET:
                break

            num = num.to(device)
            cat = cat.to(device)
            labels = labels.to(device)

            logits = model(num, cat).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Fast-fail on NaN
            if torch.isnan(loss):
                print("FATAL: NaN loss detected")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            # SWA: accumulate weights every 100 steps in the last 25% of training
            elapsed_frac = (time.time() - t0) / TIME_BUDGET
            if elapsed_frac >= swa_start_frac and step % 100 == 0:
                if swa_state is None:
                    swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                    swa_count = 1
                else:
                    for k, v in model.state_dict().items():
                        swa_state[k] += v
                    swa_count += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"  step {step:5d} | loss {loss.item():.4f} | {elapsed:.0f}s")

        if time.time() - t0 > TIME_BUDGET:
            break

    # Apply SWA averaged weights
    if swa_state is not None and swa_count > 1:
        print(f"SWA: averaged {swa_count} checkpoints")
        for k in swa_state:
            swa_state[k] /= swa_count
        model.load_state_dict(swa_state)

    training_time = time.time() - t0

    # Evaluate
    eval_start = time.time()
    metrics = evaluate(model, DATASET, device=device)
    eval_time = time.time() - eval_start
    total_time = time.time() - t0

    # Peak VRAM
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    # Print results in standard format
    print(f"\n---")
    print(f"val_logloss:      {metrics['val_logloss']:.4f}")
    print(f"val_auc:          {metrics['val_auc']:.4f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params:       {num_params}")
    print(f"num_steps:        {step}")
    print(f"dataset:          {DATASET}")


if __name__ == "__main__":
    main()
