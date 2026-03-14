"""
CTR Prediction — Training Script
=================================
This file is modified by the AI agent. It contains the model architecture,
optimizer configuration, and training loop.

Run: uv run train.py
"""
from __future__ import annotations

import sys
import time

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_config, make_dataloader, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# Hyperparameters (agent tunes these)
# ---------------------------------------------------------------------------

DATASET = "criteo"
EMBEDDING_DIM = 8
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 3e-4
DROPOUT = 0.3
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-5

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

        # Input dim = numerical features + all embeddings concatenated
        input_dim = config.num_numerical + config.num_categorical * EMBEDDING_DIM

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(DROPOUT))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        # Embed each categorical feature
        embedded = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]
        embedded = torch.cat(embedded, dim=-1)  # (batch, num_cat * emb_dim)

        # Concatenate with numerical features
        x = torch.cat([numerical, embedded], dim=-1)

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

    # Estimate total steps for cosine schedule with warmup
    steps_per_epoch = config.num_train // BATCH_SIZE
    estimated_epochs = 10  # conservative estimate for 120s budget
    total_steps = steps_per_epoch * estimated_epochs
    warmup_steps = 500
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

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
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"  step {step:5d} | loss {loss.item():.4f} | {elapsed:.0f}s")

        if time.time() - t0 > TIME_BUDGET:
            break

    training_time = time.time() - t0

    # Evaluate
    eval_start = time.time()
    metrics = evaluate(model, DATASET, device=device)
    eval_time = time.time() - eval_start
    total_time = time.time() - t0

    # Print results in standard format
    print(f"\n---")
    print(f"val_logloss:      {metrics['val_logloss']:.4f}")
    print(f"val_auc:          {metrics['val_auc']:.4f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"num_params:       {num_params}")
    print(f"num_steps:        {step}")
    print(f"dataset:          {DATASET}")


if __name__ == "__main__":
    main()
