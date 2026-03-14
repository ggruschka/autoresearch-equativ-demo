# Autoresearch: Floor Price Optimization

## Objective

Maximize **total_revenue** on validation auctions. Constraint: **fill_rate** must stay above 0.40 (don't set floors so high that nothing fills).

## The Loop

1. Read `train.py` and understand the current pricing logic
2. Decide on ONE modification (parameter tweak, new rule, different formula)
3. Edit `train.py` and commit: `git add train.py && git commit -m "description of change"`
4. Run: `uv run train.py > run.log 2>&1`
5. Extract metrics: `grep "^total_revenue:\|^fill_rate:" run.log`
6. Compare to previous best:
   - **Improved** (higher total_revenue AND fill_rate >= 0.40): Keep. Log to results.tsv.
   - **Worse, crashed, or fill_rate < 0.40**: `git reset --hard HEAD~1`. Log to results.tsv with status=regressed/crashed.
7. Repeat from step 1.

## Rules

- **Only modify `train.py`**. Never modify `prepare.py`.
- **Cannot add dependencies**. Only use what's in pyproject.toml.
- **TIME_BUDGET is 120 seconds** per iteration. Simulations run much faster (~10s) but respect the cap.
- **One change at a time**. Makes it clear what helped.
- **Log everything** to `results.tsv` (tab-separated: commit, total_revenue, fill_rate, status, description).

## Domain Hints

This is a **pricing optimization** problem. The goal is to find the sweet spot: floors high enough to extract more revenue per impression, but low enough that auctions still fill.

### Key dynamics
- Setting floor = $0 gives maximum fill rate but minimum revenue (pure second-price)
- Setting floor too high rejects most bids → zero revenue
- The optimal floor depends on the bid distribution, which varies by features
- **Second-price auctions**: the winner pays max(second bid, floor). So raising the floor above the second bid but below the first bid directly increases revenue

### Things worth trying

**Parameter tuning:**
- Adjust base floors per format (the biggest lever)
- Tweak geo/device/time multipliers
- Change publisher blend weight
- Adjust the publisher history percentile (currently 50% of avg — try median, p25, p75)

**New rules:**
- Weekend vs weekday adjustments
- More granular hour-of-day curves (not just peak/off-peak)
- Publisher-specific floor caps
- Format × geo interaction terms
- Seasonal patterns (day of week)

**Algorithm changes:**
- Use percentile-based floors (e.g., set floor at p30 of historical bids)
- Segment-specific optimization (compute optimal floor per segment)
- Multi-factor lookup table instead of multiplicative rules
- Dynamic publisher scoring based on bid variance, not just mean
- Bid density estimation — set floor where expected revenue is maximized

**Advanced:**
- Per-publisher, per-format floors (if enough data)
- Grid search over key parameters using training data
- Compute the theoretically optimal floor per auction using training bid distributions
- Revenue curve analysis: for each segment, find the floor that maximizes E[revenue]

## Output Format

`train.py` must print these lines (parsed by the evaluation harness):

```
---
total_revenue:    XXXXX.XX
fill_rate:        X.XXXX
avg_floor:        X.XX
avg_winning_bid:  X.XX
simulation_seconds: X.X
total_seconds:    X.X
```
