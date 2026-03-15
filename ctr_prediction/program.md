# Autoresearch: CTR Prediction

## Objective

Minimize **val_logloss** on the CTR prediction task. Secondary metric: maximize **val_auc**.

**The goal is simple: get the lowest val_logloss.** Since the time budget is fixed, you don't need to worry about training time — it's always 300 seconds. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is before making any changes.

## The Loop

1. Read `train.py` and understand the current state
2. Decide on a modification — hyperparameter tweaks, architecture changes, training improvements, or combinations thereof. Everything is fair game.
3. Edit `train.py` and commit: `git add train.py && git commit -m "description of change"`
4. Run: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Extract metrics: `grep "^val_logloss:\|^val_auc:" run.log`
6. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace. If it's a simple fix (typo, missing import), fix and re-run. If the idea is fundamentally broken, skip it.
7. Compare to previous best:
   - **Improved** (lower val_logloss): Keep the commit. Log to results.tsv. Update progress.png. Push.
   - **Worse**: `git reset` back to where you started. Log to results.tsv with status=discard. Update progress.png. Push.
   - **Crashed**: Log to results.tsv with status=crashed. Update progress.png. Push.
8. Repeat from step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. The loop runs until the human interrupts you, period.

**Think harder**: If you run out of ideas, don't give up. Re-read `train.py` and `prepare.py` for new angles. Try combining previous near-misses. Try more radical architectural changes. Do online research — search for papers, blog posts, and state-of-the-art techniques for CTR prediction (e.g. DCN, DeepFM, AutoInt, xDeepFM, Criteo benchmark winners) to inform new experiment ideas.

## Rules

- **Only modify `train.py`**. Never modify `prepare.py`.
- **Cannot add dependencies**. Only use what's in pyproject.toml.
- **TIME_BUDGET is 300 seconds**. Do not change it. Kill runs exceeding 10 minutes.
- **Log everything** to `results.tsv` (tab-separated — NOT comma-separated). Columns: commit (7 chars), val_logloss, val_auc, memory_gb (peak_vram_mb / 1024, rounded to .1f — use 0.0 for crashes), status (`keep`/`discard`/`crash`), description.
- **Memory**: Some VRAM increase is acceptable for meaningful val_logloss gains, but it should not blow up dramatically. If a run OOMs, treat it as a crash — revert and try a smaller version of the idea.
- **Crashes**: If a run crashes and it's easy to fix (typo, shape mismatch), fix and re-run. If you can't get it working after 2-3 attempts, give up on that idea and move on.
- **Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## Output Format

`train.py` must print these lines (parsed by the evaluation harness):

```
---
val_logloss:      X.XXXX
val_auc:          X.XXXX
training_seconds: XXX.X
total_seconds:    XXX.X
peak_vram_mb:     XXXXX.X
num_params:       XXXXX
num_steps:        XXXXX
dataset:          criteo
```
