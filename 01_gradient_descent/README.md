Gradient Descent — Visual Explainer

An interactive, from-scratch breakdown of how Gradient Descent works — covering the math, the code, and the tradeoffs between Batch GD, SGD, and Mini-batch GD.

Show Image
Live Demo →

What this is
Most ML tutorials hand you model.fit() and move on. This project goes one level deeper — deriving the gradient of MSE loss by hand, implementing all three GD variants from scratch in NumPy, and building an interactive visual explainer so the intuition actually sticks.
Built on the California Housing dataset. No PyTorch. No sklearn optimizers. Just NumPy, math, and a loss curve you can watch descend in real time.

What this covers
TopicDetailMSE loss derivationStep-by-step from L = (1/n)Σ(ŷ−y)²Partial derivativesWhy ∂L/∂w = (2/n) · Xᵀ @ err, shape-tracedWhy X.T appearsFeature-wise gradient aggregation explainedInteractive stepperWatch one GD step at a time with live mathLearning rate effectsToo small / just right / too high — chartedBatch GDExact gradient, slow on large dataStochastic GD1 sample per update, noisy but fastMini-batch GD32–256 samples, GPU-friendly, industry standard

Key insight
Gradient Descent does not find the best model by trying random weights. It starts somewhere, measures how wrong it is (the loss), then asks: which direction do I move each weight to reduce that wrongness the most? That direction is the negative gradient. The learning rate controls how big a step to take. Do this enough times and you converge to weights that make good predictions.
The reason X.T @ error appears in the gradient is not magic — it is a matrix shorthand for summing up how much each feature contributed to the prediction error, across all training samples, in a single vectorized operation. The shape forces it: X.T is (n_features, n_samples), error is (n_samples,), so the result is (n_features,) — one gradient value per weight, exactly what we need.

Project structure
gradient-descent-visual/
├── index.html                  # interactive visual explainer (live on GitHub Pages)
├── notebook/
│   └── gradient_descent.ipynb  # full NumPy implementation with outputs
├── assets/
│   └── preview.png             # screenshot for README
└── README.md

Run the notebook locally
bash# clone the repo
git clone https://github.com/yourusername/gradient-descent-visual
cd gradient-descent-visual

# install dependencies
pip install numpy matplotlib scikit-learn jupyter

# launch notebook
jupyter notebook notebook/gradient_descent.ipynb

Implementation — core functions
pythondef predict(X, w, b):
    return X @ w + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradients(X, w, b, y):
    err = predict(X, w, b) - y          # (n_samples,)
    dw  = (2/len(X)) * (X.T @ err)     # (n_features,)
    db  = (2/len(X)) * np.sum(err)     # scalar
    return dw, db

def vanilla_gd(X, y, lr=0.05, epochs=500):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for epoch in range(epochs):
        dw, db = gradients(X, w, b, y)
        w -= lr * dw
        b -= lr * db
        losses.append(mse(y, predict(X, w, b)))
    return w, b, losses

Results on California Housing
Using a single feature (AveRooms), standardized inputs, normalized target:
VariantFinal MSEEpochsUpdates/epochBatch GD~0.875001SGD~0.88500n (≈12,384)Mini-batch GD~0.87500n/32 (≈387)
Single-feature linear regression has a hard floor around 0.85–0.90 MSE on this dataset — the relationship between AveRooms and price alone is weak. Adding more features would push this lower.

Stack

NumPy — all math and gradient computation
scikit-learn — dataset loading, train/test split, StandardScaler
Matplotlib — training plots in notebook
Chart.js — interactive loss curves in the web explainer
Vanilla JS + HTML/CSS — no framework, fully self-contained


Why I built this
Understanding GD at the derivative level matters for debugging real models — knowing why a loss curve oscillates, why batch size affects convergence speed, and why learning rate is the most sensitive hyperparameter are things you only internalize by implementing it from scratch. This project is that implementation, made visual.
