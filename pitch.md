# Autoresearch: Autonomous Algorithm Optimization

**TL;DR** — An AI agent modifies code, runs short experiments, keeps improvements, discards regressions, and repeats overnight. It works for **any problem with a clear metric** — neural nets, traditional ML, or pure algorithmic.

---

## What is this?

A loop:
1. AI reads the current algorithm/model code
2. Makes a targeted modification (hyperparameters, architecture, logic)
3. Runs a short experiment (2–5 minutes)
4. Compares the result metric to the previous best
5. **Keeps** improvements (git commit), **discards** regressions (git reset)
6. Repeats — indefinitely

The key insight: **the AI doesn't need to be right every time**. It just needs to be right *often enough*. Bad ideas get discarded automatically. Good ideas compound.

## Inspiration

This pattern was introduced by Andrej Karpathy ([autoresearch repo](https://github.com/karpathy/autoresearch)) for LLM pretraining. His system improved a GPT model overnight by autonomously iterating on architecture and hyperparameters.

## Why it fits Equativ

Equativ has several problems where:
- There's a **clear metric** (revenue, fill rate, latency, prediction accuracy)
- The current solution has **tunable parameters or algorithmic choices**
- Running an experiment takes **minutes, not hours**

Concrete candidates:
| Use Case | Metric | Type |
|---|---|---|
| **Floor price optimization** | Revenue per impression | Algorithmic (rules + parameters) |
| **CTR prediction** | Log-loss / AUC | Neural network or ML model |
| **Traffic shaping / call limiter** | Revenue vs. infrastructure cost | Algorithmic |
| **Bid request filtering** | Fill rate at latency target | Algorithmic |

## Demo: Two Problems, One Pattern

We built two working demos to show this isn't just about neural networks:

### Demo 1: CTR Prediction (Neural Network)
- **Problem**: Predict whether a user clicks an ad
- **Approach**: MLP with learned embeddings for categorical features
- **Metric**: Validation log-loss (lower = better)
- **What the agent optimizes**: Embedding dimensions, hidden layer sizes, learning rate, dropout, architecture

### Demo 2: Floor Price Optimization (No ML — Pure Algorithmic)
- **Problem**: Set minimum bid prices to maximize revenue
- **Approach**: Rule-based pricing with multipliers for format, time, geo, device
- **Metric**: Total revenue on held-out auctions (higher = better)
- **What the agent optimizes**: Base prices, multipliers, blending strategies, new rules

**Same loop. Same tools. Completely different problem types.**

## What's Needed for a Real Equativ Pilot

1. **Historical data export** — enough to build a representative eval set
2. **Evaluation harness** — a script that scores a candidate solution against the data
3. **Well-defined metric** — what "better" means, quantitatively
4. **Compute** — a machine with GPU (for ML) or just CPU (for algorithmic)

The AI agent and loop infrastructure already exist. The hard part is defining the right eval — which is valuable engineering work regardless.

## Proposed Next Steps

1. **Pick one real use case** — ideally one where we already have data and a metric
2. **Build the eval harness** together (1–2 days of engineering work)
3. **Run the agent overnight** and review results in the morning
4. **Evaluate**: Did it find improvements? Are they deployable?

---

*Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern. Demo repo available for hands-on exploration.*
