"""Generate progress.png from results.tsv"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Parse results.tsv
experiments = []
with open('results.tsv') as f:
    header = f.readline()
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        commit, val_logloss, val_auc, mem, status, desc = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), parts[4], parts[5]
        experiments.append({
            'idx': i,
            'commit': commit,
            'val_logloss': val_logloss,
            'status': status,
            'description': desc,
        })

# Separate kept vs discarded
kept = [e for e in experiments if e['status'] == 'keep']
discarded = [e for e in experiments if e['status'] != 'keep']

# Compute running best
running_best = []
best_so_far = float('inf')
for e in experiments:
    if e['status'] == 'keep' and e['val_logloss'] < best_so_far:
        best_so_far = e['val_logloss']
    running_best.append(best_so_far)

# Find indices where best improved (for annotations)
best_improvements = []
prev_best = float('inf')
for e in experiments:
    if e['status'] == 'keep' and e['val_logloss'] < prev_best:
        best_improvements.append(e)
        prev_best = e['val_logloss']

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

# Discarded experiments (gray dots)
ax.scatter(
    [e['idx'] for e in discarded],
    [e['val_logloss'] for e in discarded],
    c='#cccccc', s=40, alpha=0.6, label='Discarded', zorder=2
)

# Kept experiments (green dots)
ax.scatter(
    [e['idx'] for e in kept],
    [e['val_logloss'] for e in kept],
    c='#2ecc71', s=60, edgecolors='white', linewidths=0.5, label='Kept', zorder=3
)

# Running best line
best_x = []
best_y = []
current_best = float('inf')
for e in experiments:
    if e['status'] == 'keep' and e['val_logloss'] < current_best:
        current_best = e['val_logloss']
        best_x.append(e['idx'])
        best_y.append(current_best)
# Extend to last experiment
best_x.append(experiments[-1]['idx'])
best_y.append(current_best)
ax.step(best_x, best_y, where='post', c='#2ecc71', linewidth=2, alpha=0.8, label='Running best', zorder=1)

# Annotate kept improvements
for e in best_improvements:
    # Truncate long descriptions
    desc = e['description']
    if len(desc) > 45:
        desc = desc[:42] + '...'
    ax.annotate(
        desc,
        (e['idx'], e['val_logloss']),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=7,
        color='#555555',
        rotation=25,
        ha='left',
    )

n_kept = len(kept)
n_total = len(experiments)
ax.set_title(f'Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements', fontsize=14, fontweight='bold')
ax.set_xlabel('Experiment #', fontsize=12)
ax.set_ylabel('Validation LogLoss (lower is better)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('progress.png', dpi=170, bbox_inches='tight')
print(f"Saved progress.png ({n_total} experiments, {n_kept} kept)")
