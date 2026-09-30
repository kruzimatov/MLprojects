# ML Projects — Classic Machine Learning from Scratch

Linear Regression implemented three ways: closed-form solution, gradient descent from scratch, and full sklearn workflow — each on a real dataset.

## Projects

### 01 — Normal Equation (Credit Dataset)
Predicting credit card balance from 6 customer features (ISLR Credit, 400 samples).
- Normal Equation `θ = (XᵀX)⁻¹Xᵀy` implemented from scratch with NumPy
- MSE, RMSE, MAE, R² coded by hand, verified against sklearn
- Feature importance analysis, multicollinearity (condition number), 3D MSE surface + contour plots
- **Test R² = 0.86**

### 02 — Gradient Descent (Concrete Dataset)
Predicting concrete compressive strength from 8 mix components (UCI, 1030 samples).
- `LinearRegressionGD` class: batch, SGD, and mini-batch modes from scratch
- Learning rate experiments, convergence analysis vs Normal Equation
- Momentum and early stopping implementations
- Feature engineering: `log(age)` transformation
- **R² improved from 0.63 → 0.83 with log transform**

### 03 — sklearn Workflow (Palmer Penguins)
End-to-end ML pipeline: messy data → clean prediction (342 samples, mixed types).
- Leakage-safe preprocessing: train-only statistics for scaling and imputation
- One-hot encoding, `predict_one()` for new data with unseen categories
- `Pipeline` + `ColumnTransformer` for reproducible workflow
- Simpson's Paradox analysis (bill_depth correlation flip)
- **Test R² = 0.88, RMSE = 287g**

### 04 — Walmart Sales (Linear Regression from Scratch)
Predicting weekly sales from store, time and economic features (Walmart Store Sales dataset).
- Time-based train/test split (no random shuffling — respects chronological order)
- Normal Equation and gradient descent implemented from scratch, compared against sklearn
- EDA: correlation heatmap, time patterns, feature relationships
- Feature scaling, residual analysis, feature importance by weight magnitude

## Tech Stack
Python, NumPy, pandas, matplotlib, seaborn, scikit-learn

## Author
[@kruzimatov](https://github.com/kruzimatov)
