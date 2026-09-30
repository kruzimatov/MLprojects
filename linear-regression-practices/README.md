# Linear Regression — Practices

Applying the fundamentals (normal equation, gradient descent, sklearn) to a full, messy, real-world dataset — end to end, from raw CSV to a working model.

## 04 — Walmart Sales
Predicting weekly sales from store, time and economic features (Walmart Store Sales dataset).
- Time-based train/test split (no random shuffling — respects chronological order, prevents leaking future data into training)
- Normal Equation and gradient descent implemented from scratch, compared against sklearn
- EDA: correlation heatmap, time patterns, feature relationships
- Feature scaling, residual analysis, feature importance by weight magnitude

Full walkthrough: [04-walmart-sales/EXPLANATION.md](04-walmart-sales/EXPLANATION.md)

## Why this folder is separate from fundamentals

The fundamentals folder teaches one technique per notebook on a clean dataset. This folder is the opposite: one notebook, every technique combined, on a dataset with real problems — missing values, mixed scales, dates as strings, categorical stores, seasonal patterns. This is what applying the fundamentals actually looks like once the dataset stops cooperating.
