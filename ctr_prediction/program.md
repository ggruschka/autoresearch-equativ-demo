# Autoresearch: CTR Prediction

## Objective

Minimize **val_logloss** on the CTR prediction task. Secondary metric: maximize **val_auc**.

## The Loop

1. Read `train.py` and understand the current state
2. Decide on ONE modification (hyperparameter tweak, architecture change, or training improvement)
3. Edit `train.py` and commit: `git add train.py && git commit -m "description of change"`
4. Run: `uv run train.py > run.log 2>&1`
5. Extract metrics: `grep "^val_logloss:\|^val_auc:" run.log`
6. Compare to previous best:
   - **Improved** (lower val_logloss): Keep the commit. Log to results.tsv.
   - **Worse or crashed**: `git reset --hard HEAD~1`. Log to results.tsv with status=regressed/crashed.
7. Repeat from step 1.

## Rules

- **Only modify `train.py`**. Never modify `prepare.py`.
- **Cannot add dependencies**. Only use what's in pyproject.toml.
- **TIME_BUDGET is 120 seconds**. Do not change it. Kill runs exceeding 5 minutes.
- **One change at a time**. Makes it clear what helped.
- **Log everything** to `results.tsv` (tab-separated: commit, val_logloss, val_auc, status, description).

## Domain Hints

Things worth trying (in rough order of expected impact):

### Hyperparameters
- Embedding dimension (current: 8). Try 16, 32 — larger embeddings capture richer feature interactions
- Hidden layer sizes and depth. Try [128, 64], [256, 128, 64], etc.
- Learning rate (current: 1e-3). Try 3e-4, 5e-4, 2e-3
- Batch size (current: 1024). Try 512, 2048, 4096
- Dropout (current: 0.1). Try 0.0, 0.2, 0.3
- Weight decay. Try 1e-4, 1e-6

### Architecture
- Add batch normalization between layers
- Try different activations (GELU, SiLU/Swish instead of ReLU)
- Feature interaction layers (e.g., element-wise products of embeddings)
- Skip/residual connections for deeper networks
- Separate numerical feature processing (small MLP before concatenation)

### Training
- Learning rate scheduling (cosine decay, warmup)
- Gradient clipping
- Different optimizers (Adam vs AdamW, try different betas)
- Label smoothing

### Advanced
- DeepFM-style: add a factorization machine component alongside the MLP
- Cross-network (DCN): explicit feature crossing
- Attention over feature embeddings
- Per-feature embedding dimensions (larger for high-cardinality features)

## Output Format

`train.py` must print these lines (parsed by the evaluation harness):

```
---
val_logloss:      X.XXXX
val_auc:          X.XXXX
training_seconds: XXX.X
total_seconds:    XXX.X
num_params:       XXXXX
num_steps:        XXXXX
dataset:          criteo
```
