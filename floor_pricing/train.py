"""
Floor Price Optimization — Pricing Algorithm
==============================================
This file is modified by the AI agent. It contains a rule-based pricing
algorithm with tunable parameters. No neural networks, no gradient descent.

Run: uv run train.py
"""
from __future__ import annotations

import time
from collections import defaultdict

from prepare import load_data, load_train_data, evaluate, AuctionRequest

# ---------------------------------------------------------------------------
# Pricing Parameters (agent tunes these)
# ---------------------------------------------------------------------------

# Base floor prices by ad format ($CPM)
BASE_FLOOR_DISPLAY = 0.50
BASE_FLOOR_VIDEO = 2.00
BASE_FLOOR_NATIVE = 0.80

# Time-of-day multipliers
PEAK_HOURS = [11, 12, 13, 14, 19, 20, 21]
PEAK_MULTIPLIER = 1.3
OFFPEAK_MULTIPLIER = 0.8

# Geo tier adjustments
GEO_TIER1_MULT = 1.2
GEO_TIER2_MULT = 1.0
GEO_TIER3_MULT = 0.7

# Device adjustments
DEVICE_DESKTOP_MULT = 1.1
DEVICE_MOBILE_MULT = 1.0
DEVICE_TABLET_MULT = 0.9

# Publisher performance adjustment
USE_PUBLISHER_HISTORY = True
PUBLISHER_BLEND_WEIGHT = 0.3  # blend historical avg with base

# ---------------------------------------------------------------------------
# Publisher history (computed from training data)
# ---------------------------------------------------------------------------

publisher_stats: dict[int, float] = {}


def compute_publisher_stats():
    """Compute average winning bid per publisher from training data."""
    global publisher_stats
    train_data = load_train_data()

    pub_bids = defaultdict(list)
    for req in train_data.requests:
        if req.bids:
            pub_bids[req.publisher_id].append(max(req.bids))

    publisher_stats = {
        pub_id: sum(bids) / len(bids)
        for pub_id, bids in pub_bids.items()
        if len(bids) >= 10  # only use publishers with enough data
    }


# ---------------------------------------------------------------------------
# Pricing Function
# ---------------------------------------------------------------------------

def compute_floor_price(req: AuctionRequest) -> float:
    """Compute the floor price for an auction request.

    This is the function the agent optimizes by modifying parameters
    and logic above.
    """
    # Base floor by format
    if req.ad_format == "display":
        floor = BASE_FLOOR_DISPLAY
    elif req.ad_format == "video":
        floor = BASE_FLOOR_VIDEO
    elif req.ad_format == "native":
        floor = BASE_FLOOR_NATIVE
    else:
        floor = BASE_FLOOR_DISPLAY

    # Time-of-day adjustment
    if req.hour_of_day in PEAK_HOURS:
        floor *= PEAK_MULTIPLIER
    else:
        floor *= OFFPEAK_MULTIPLIER

    # Geo tier adjustment
    if req.geo_tier == "tier1":
        floor *= GEO_TIER1_MULT
    elif req.geo_tier == "tier2":
        floor *= GEO_TIER2_MULT
    elif req.geo_tier == "tier3":
        floor *= GEO_TIER3_MULT

    # Device adjustment
    if req.device_type == "desktop":
        floor *= DEVICE_DESKTOP_MULT
    elif req.device_type == "mobile":
        floor *= DEVICE_MOBILE_MULT
    elif req.device_type == "tablet":
        floor *= DEVICE_TABLET_MULT

    # Publisher historical performance blend
    if USE_PUBLISHER_HISTORY and req.publisher_id in publisher_stats:
        hist_avg = publisher_stats[req.publisher_id]
        # Use a fraction of historical average as floor hint
        hist_floor = hist_avg * 0.5  # set floor at 50% of avg winning bid
        floor = (1 - PUBLISHER_BLEND_WEIGHT) * floor + PUBLISHER_BLEND_WEIGHT * hist_floor

    return max(floor, 0.01)  # minimum floor of $0.01


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # Compute publisher stats from training data
    if USE_PUBLISHER_HISTORY:
        print("Computing publisher statistics from training data...")
        compute_publisher_stats()
        print(f"  {len(publisher_stats)} publishers with sufficient history")

    # Run evaluation on validation set
    print("Running auction simulation on validation data...")
    sim_start = time.time()
    metrics = evaluate(compute_floor_price)
    sim_time = time.time() - sim_start
    total_time = time.time() - t0

    # Print results in standard format
    print(f"\n---")
    print(f"total_revenue:    {metrics['total_revenue']:.2f}")
    print(f"fill_rate:        {metrics['fill_rate']:.4f}")
    print(f"avg_floor:        {metrics['avg_floor']:.2f}")
    print(f"avg_winning_bid:  {metrics['avg_winning_bid']:.2f}")
    print(f"simulation_seconds: {sim_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")


if __name__ == "__main__":
    main()
