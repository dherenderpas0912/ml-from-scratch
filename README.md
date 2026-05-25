# ML From Scratch

> Every major ML algorithm — derived mathematically, implemented in NumPy, and explained with interactive visuals. Built for deep understanding, not just `model.fit()`.

**[Live Portfolio →](https://dherenderpas0912.github.io/ml-from-scratch)**

---

## Why this exists

Most ML practitioners can call a sklearn function. Fewer can explain why the gradient of MSE loss involves `X.T`, or what actually happens inside a decision tree split. This repo is the gap between using ML and understanding it — every algorithm built from the math up, with visuals that make the intuition stick.

---

## Topics

| # | Topic | Notebook | Live Visual | Key concept covered |
|---|---|---|---|---|
| 01 | Gradient Descent | [notebook](docs/01_gradient_descent/notebook.ipynb) | [open →](https://dherenderpas0912.github.io/ml-from-scratch/01_gradient_descent/index.html) | ∂L/∂w derivation, Batch vs SGD vs Mini-batch |
| 02 | Linear Regression | coming soon | coming soon | OLS solution vs GD, R² |
| 03 | Logistic Regression | coming soon | coming soon | Sigmoid, BCE loss, decision boundary |
| 04 | Regularization | coming soon | coming soon | L1 vs L2, bias-variance tradeoff |
| 05 | Decision Trees | coming soon | coming soon | Information gain, Gini impurity |
| 06 | Random Forest | coming soon | coming soon | Bagging, feature sampling, OOB error |
| 07 | XGBoost | coming soon | coming soon | Boosting, residual fitting |
| 08 | Neural Networks | coming soon | coming soon | Forward pass, backprop |
| 09 | K-Means | coming soon | coming soon | Cluster assignment, centroid update |
| 10 | PCA | coming soon | coming soon | Eigenvectors, explained variance |

---

## Roadmap

```
Phase 1 — Foundations
  ✅ 01. Gradient Descent
  ⬜ 02. Linear Regression
  ⬜ 03. Logistic Regression
  ⬜ 04. Regularization (L1 / L2)

Phase 2 — Tree models
  ⬜ 05. Decision Trees
  ⬜ 06. Random Forest
  ⬜ 07. XGBoost

Phase 3 — Neural networks
  ⬜ 08. Neural Network (1 hidden layer)
  ⬜ 09. Backpropagation

Phase 4 — Unsupervised
  ⬜ 10. K-Means Clustering
```

---

## Run any notebook

```bash
git clone https://github.com/dherenderpas0912/ml-from-scratch
cd ml-from-scratch
pip install numpy matplotlib scikit-learn jupyter
jupyter notebook docs/01_gradient_descent/notebook.ipynb
```

---

## Stack

- **NumPy** — all math and model implementation
- **scikit-learn** — datasets, preprocessing, benchmarking
- **Matplotlib** — notebook plots
- **Chart.js + vanilla JS** — interactive web explainers
- **GitHub Pages** — hosting, zero infra
