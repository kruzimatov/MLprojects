# Linear Regression — Fundamentals

Three exercises, each isolating one core technique of linear regression. Same underlying model (`y = Xθ`), three different ways to find `θ`, three different datasets to force different challenges (multicollinearity, non-linearity, mixed data types).

## 01 — Normal Equation (Credit Dataset)
Predicting credit card balance from 6 customer features (ISLR Credit, 400 samples).
- Normal Equation `θ = (XᵀX)⁻¹Xᵀy` implemented from scratch with NumPy
- MSE, RMSE, MAE, R² coded by hand, verified against sklearn
- Feature importance analysis, multicollinearity (condition number), 3D MSE surface + contour plots
- **Test R² = 0.86**

Full walkthrough: [01-normal-equation/EXPLANATION.md](01-normal-equation/EXPLANATION.md)

## 02 — Gradient Descent (Concrete Dataset)
Predicting concrete compressive strength from 8 mix components (UCI, 1030 samples).
- `LinearRegressionGD` class: batch, SGD, and mini-batch modes from scratch
- Learning rate experiments, convergence analysis vs Normal Equation
- Momentum and early stopping implementations
- Feature engineering: `log(age)` transformation
- **R² improved from 0.63 → 0.83 with log transform**

Full walkthrough: [02-gradient-descent/EXPLANATION.md](02-gradient-descent/EXPLANATION.md)

## 03 — sklearn Workflow (Palmer Penguins)
End-to-end ML pipeline: messy data → clean prediction (342 samples, mixed types).
- Leakage-safe preprocessing: train-only statistics for scaling and imputation
- One-hot encoding, `predict_one()` for new data with unseen categories
- `Pipeline` + `ColumnTransformer` for reproducible workflow
- Simpson's Paradox analysis (bill_depth correlation flip)
- **Test R² = 0.88, RMSE = 287g**

Full walkthrough: [03-sklearn-workflow/EXPLANATION.md](03-sklearn-workflow/EXPLANATION.md)

## Why three ways to do the same thing?

- **Normal Equation** is exact and fast for small feature counts, but does not scale (matrix inversion is `O(n³)`) and breaks down with near-duplicate (collinear) features.
- **Gradient Descent** scales to any size, is the same algorithm used to train neural networks, and lets you see *how* learning happens (loss curve, convergence, learning rate tradeoffs) — not just the final answer.
- **sklearn Workflow** is what you actually use in practice: it wraps both ideas behind a tested, leakage-safe, production-ready API.

Understanding all three means you understand what sklearn is doing under the hood, not just how to call `.fit()`.
