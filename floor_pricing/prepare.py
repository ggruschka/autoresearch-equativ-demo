"""
Floor Price Optimization — Synthetic Auction Data & Evaluation Harness
=======================================================================
DO NOT MODIFY THIS FILE. The agent modifies train.py only.

Generates synthetic programmatic advertising auction data and provides
an evaluation function that simulates auction outcomes given a pricing function.

Usage:
    uv run prepare.py          # Generate and cache auction data
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch-equativ-demo/floor"))
TIME_BUDGET = 120  # seconds per experiment (though simulations run much faster)

NUM_TRAIN = 200_000
NUM_VAL = 50_000
NUM_PUBLISHERS = 50
SEED_TRAIN = 42
SEED_VAL = 999

AD_FORMATS = ["display", "video", "native"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
GEO_TIERS = ["tier1", "tier2", "tier3"]

# ---------------------------------------------------------------------------
# Auction data structure
# ---------------------------------------------------------------------------

@dataclass
class AuctionRequest:
    """A single ad auction request with its features and bids."""
    publisher_id: int
    ad_format: str
    device_type: str
    hour_of_day: int
    day_of_week: int
    geo_tier: str
    bids: list[float]  # potential bids from advertisers


@dataclass
class AuctionData:
    """Full dataset of auction requests."""
    requests: list[AuctionRequest]

    def __len__(self):
        return len(self.requests)


# ---------------------------------------------------------------------------
# Bid distribution parameters (vary by features to make it learnable)
# ---------------------------------------------------------------------------

# Base CPM by format (log-normal mu parameter)
FORMAT_MU = {"display": -0.5, "video": 0.8, "native": 0.1}
FORMAT_SIGMA = {"display": 0.8, "video": 0.7, "native": 0.6}

# Geo tier multipliers on mu
GEO_MU_OFFSET = {"tier1": 0.3, "tier2": 0.0, "tier3": -0.4}

# Device adjustments
DEVICE_MU_OFFSET = {"desktop": 0.1, "mobile": 0.0, "tablet": -0.1}

# Hour-of-day effect (sinusoidal — peaks at lunch and evening)
def _hour_effect(hour: int) -> float:
    import math
    return 0.15 * math.sin(2 * math.pi * (hour - 6) / 24) + \
           0.10 * math.sin(2 * math.pi * (hour - 12) / 12)

# Publisher quality (deterministic from ID)
def _publisher_quality(pub_id: int) -> float:
    """Returns a quality factor in [0.5, 1.5] — some publishers attract better bids."""
    rng = np.random.RandomState(pub_id * 7 + 13)
    return 0.5 + rng.rand() * 1.0


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_auctions(n: int, seed: int) -> AuctionData:
    """Generate n synthetic auction requests with realistic bid distributions."""
    rng = np.random.RandomState(seed)
    requests = []

    for _ in range(n):
        pub_id = rng.randint(0, NUM_PUBLISHERS)
        ad_format = AD_FORMATS[rng.randint(0, len(AD_FORMATS))]
        device = DEVICE_TYPES[rng.randint(0, len(DEVICE_TYPES))]
        hour = rng.randint(0, 24)
        dow = rng.randint(0, 7)
        geo = GEO_TIERS[rng.randint(0, len(GEO_TIERS))]

        # Number of potential bidders (Poisson, mean depends on format)
        mean_bidders = {"display": 4.0, "video": 2.5, "native": 3.0}[ad_format]
        num_bidders = max(1, rng.poisson(mean_bidders))

        # Bid distribution parameters
        mu = FORMAT_MU[ad_format] + GEO_MU_OFFSET[geo] + DEVICE_MU_OFFSET[device]
        mu += _hour_effect(hour)
        mu += 0.2 * (_publisher_quality(pub_id) - 1.0)
        sigma = FORMAT_SIGMA[ad_format]

        # Generate bids (log-normal)
        bids = rng.lognormal(mean=mu, sigma=sigma, size=num_bidders).tolist()
        # Clip to realistic range
        bids = [max(0.01, min(b, 50.0)) for b in bids]

        requests.append(AuctionRequest(
            publisher_id=pub_id,
            ad_format=ad_format,
            device_type=device,
            hour_of_day=hour,
            day_of_week=dow,
            geo_tier=geo,
            bids=bids,
        ))

    return AuctionData(requests=requests)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _save_auctions(path: Path, data: AuctionData) -> None:
    """Save auction data as numpy arrays for fast loading."""
    path.mkdir(parents=True, exist_ok=True)
    n = len(data)

    pub_ids = np.array([r.publisher_id for r in data.requests], dtype=np.int32)
    formats = np.array([AD_FORMATS.index(r.ad_format) for r in data.requests], dtype=np.int32)
    devices = np.array([DEVICE_TYPES.index(r.device_type) for r in data.requests], dtype=np.int32)
    hours = np.array([r.hour_of_day for r in data.requests], dtype=np.int32)
    dows = np.array([r.day_of_week for r in data.requests], dtype=np.int32)
    geos = np.array([GEO_TIERS.index(r.geo_tier) for r in data.requests], dtype=np.int32)

    # Store bids as variable-length — save flat array + offsets
    all_bids = []
    offsets = [0]
    for r in data.requests:
        all_bids.extend(r.bids)
        offsets.append(len(all_bids))

    np.save(path / "pub_ids.npy", pub_ids)
    np.save(path / "formats.npy", formats)
    np.save(path / "devices.npy", devices)
    np.save(path / "hours.npy", hours)
    np.save(path / "dows.npy", dows)
    np.save(path / "geos.npy", geos)
    np.save(path / "bids.npy", np.array(all_bids, dtype=np.float64))
    np.save(path / "bid_offsets.npy", np.array(offsets, dtype=np.int64))


def _load_auctions(path: Path) -> AuctionData:
    """Load cached auction data."""
    pub_ids = np.load(path / "pub_ids.npy")
    formats = np.load(path / "formats.npy")
    devices = np.load(path / "devices.npy")
    hours = np.load(path / "hours.npy")
    dows = np.load(path / "dows.npy")
    geos = np.load(path / "geos.npy")
    bids = np.load(path / "bids.npy")
    offsets = np.load(path / "bid_offsets.npy")

    requests = []
    for i in range(len(pub_ids)):
        req_bids = bids[offsets[i]:offsets[i+1]].tolist()
        requests.append(AuctionRequest(
            publisher_id=int(pub_ids[i]),
            ad_format=AD_FORMATS[formats[i]],
            device_type=DEVICE_TYPES[devices[i]],
            hour_of_day=int(hours[i]),
            day_of_week=int(dows[i]),
            geo_tier=GEO_TIERS[geos[i]],
            bids=req_bids,
        ))

    return AuctionData(requests=requests)


def _is_cached() -> bool:
    return (CACHE_DIR / "train" / "pub_ids.npy").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare() -> tuple[AuctionData, AuctionData]:
    """Generate and cache auction data. Returns (train_data, val_data)."""
    if _is_cached():
        print(f"Auction data already cached at {CACHE_DIR}")
        train = _load_auctions(CACHE_DIR / "train")
        val = _load_auctions(CACHE_DIR / "val")
        print(f"  Train: {len(train)} auctions | Val: {len(val)} auctions")
        return train, val

    print("Generating synthetic auction data...")

    print(f"  Generating {NUM_TRAIN} training auctions...")
    train = _generate_auctions(NUM_TRAIN, SEED_TRAIN)

    print(f"  Generating {NUM_VAL} validation auctions...")
    val = _generate_auctions(NUM_VAL, SEED_VAL)

    _save_auctions(CACHE_DIR / "train", train)
    _save_auctions(CACHE_DIR / "val", val)
    print(f"  Cached to {CACHE_DIR}")

    return train, val


def load_data() -> tuple[AuctionData, AuctionData]:
    """Load cached auction data (must call prepare first)."""
    if not _is_cached():
        raise RuntimeError("Auction data not prepared. Run: uv run prepare.py")
    train = _load_auctions(CACHE_DIR / "train")
    val = _load_auctions(CACHE_DIR / "val")
    return train, val


def load_train_data() -> AuctionData:
    """Load only training data (for computing publisher statistics etc.)."""
    if not _is_cached():
        raise RuntimeError("Auction data not prepared. Run: uv run prepare.py")
    return _load_auctions(CACHE_DIR / "train")


def simulate_auction(bids: list[float], floor_price: float) -> float:
    """Simulate a second-price auction with a floor price.

    Returns revenue (0 if no bids meet the floor).
    Rules:
    - Only bids >= floor_price participate
    - Winner pays max(second_highest_eligible_bid, floor_price)
    - If only one eligible bid, winner pays floor_price
    - If no eligible bids, revenue = 0
    """
    eligible = [b for b in bids if b >= floor_price]
    if not eligible:
        return 0.0
    eligible.sort(reverse=True)
    if len(eligible) == 1:
        return floor_price
    else:
        # Second-price: winner pays max(second bid, floor)
        return max(eligible[1], floor_price)


def evaluate(pricing_fn: Callable[[AuctionRequest], float],
             data: AuctionData | None = None) -> dict[str, float]:
    """Evaluate a pricing function on validation auctions.

    Args:
        pricing_fn: function(AuctionRequest) -> float (the floor price)
        data: auction data to evaluate on. If None, loads validation data.

    Returns dict with:
        total_revenue: sum of revenue across all auctions (higher is better)
        fill_rate: fraction of auctions with at least one eligible bid
        avg_floor: average floor price set
        avg_winning_bid: average winning bid (among filled auctions)
    """
    if data is None:
        _, data = load_data()

    total_revenue = 0.0
    filled = 0
    total_floor = 0.0
    total_winning_bid = 0.0

    for req in data.requests:
        floor = pricing_fn(req)
        total_floor += floor

        revenue = simulate_auction(req.bids, floor)
        total_revenue += revenue

        if revenue > 0:
            filled += 1
            # The winning bid is the highest eligible bid
            eligible = [b for b in req.bids if b >= floor]
            total_winning_bid += max(eligible)

    n = len(data)
    fill_rate = filled / n if n > 0 else 0.0
    avg_floor = total_floor / n if n > 0 else 0.0
    avg_winning_bid = total_winning_bid / filled if filled > 0 else 0.0

    return {
        "total_revenue": round(total_revenue, 2),
        "fill_rate": round(fill_rate, 4),
        "avg_floor": round(avg_floor, 2),
        "avg_winning_bid": round(avg_winning_bid, 2),
    }


# ---------------------------------------------------------------------------
# Dataset statistics (for understanding the data)
# ---------------------------------------------------------------------------

def print_stats(data: AuctionData, label: str = "Data") -> None:
    """Print summary statistics about auction data."""
    n = len(data)
    print(f"\n{label} Statistics ({n} auctions):")

    # Format distribution
    format_counts = {}
    for r in data.requests:
        format_counts[r.ad_format] = format_counts.get(r.ad_format, 0) + 1
    print(f"  Formats: {', '.join(f'{k}: {v/n:.1%}' for k, v in sorted(format_counts.items()))}")

    # Geo distribution
    geo_counts = {}
    for r in data.requests:
        geo_counts[r.geo_tier] = geo_counts.get(r.geo_tier, 0) + 1
    print(f"  Geo:     {', '.join(f'{k}: {v/n:.1%}' for k, v in sorted(geo_counts.items()))}")

    # Bid statistics
    all_bids = [b for r in data.requests for b in r.bids]
    print(f"  Bids per auction: mean={np.mean([len(r.bids) for r in data.requests]):.1f}")
    print(f"  Bid values: mean=${np.mean(all_bids):.2f}, median=${np.median(all_bids):.2f}, "
          f"p90=${np.percentile(all_bids, 90):.2f}, max=${np.max(all_bids):.2f}")

    # Baseline: what revenue do you get with floor=0 (pure second-price)?
    zero_revenue = sum(simulate_auction(r.bids, 0.0) for r in data.requests)
    print(f"  Baseline revenue (floor=$0): ${zero_revenue:,.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val = prepare()
    print_stats(train, "Training")
    print_stats(val, "Validation")
    print(f"\nReady for optimization!")
