# Machine Learning Assignment — From Linear Models to Kernels and Trees

This assignment takes a hands-on tour from classic linear models to margin-based methods and tree ensembles. It is implemented with core algorithms **from scratch** using **NumPy + Matplotlib** (with minimal, explicit exceptions for data splitting and baselines), visualize their behavior, and compare them with simple references.

## 🧭 What to build

### Q1 — Conceptual Warm-up (2 pts)

Short theory questions (e.g., XOR with a perceptron, why non-linear hidden layers fix it).
Write clear, concise markdown answers in the notebook.

---

### Q2 — Regression (15 pts)

**Part A: Linear Regression (Closed-form & Gradient Descent)**
Implement:

* Normal Equation: $\theta = (X^\top X)^{-1}X^\top y$
* Batch Gradient Descent

**Requirements**

* Dataset: $y=3x+5+\epsilon$, 100 points, 80/20 split
* **Both methods** must achieve **test MSE < 0.65**
* Plot: data + both fitted lines
* Comment on convergence speed/accuracy

**Part B: Logistic Regression (Binary Classification)**
Implement:

* Sigmoid, cross-entropy loss, gradient descent
* Linear decision boundary visualization

**Requirements**

* Simple 2-D synthetic data (label by sign of $x_1 + x_2$)
* **100% test accuracy**
* Plot: decision boundary contour at 0.5

**Part C: Support Vector Regression (SVR, ε-insensitive, RBF kernel)**
Implement:

* RBF kernel matrix
* Simplified dual over $\beta=\alpha-\alpha^*$ (projected subgradient ascent)
* Prediction: $f(x) = \sum_j \beta_j K(x_j,x) + b$

**Requirements**

* Dataset: $y=\sin(x)+\epsilon$ (non-linear)
* **MSE_svr < MSE_lr** and **MSE_svr < 0.05**
* Plot: linear regression vs SVR curves

---

### Q3 — Support Vector Machines (25 pts)

**Part A: Primal Soft-Margin Linear SVM (subgradient)**

* Objective: $\tfrac12\lVert w\rVert^2 + C\sum_i \max(0,1-y_i(w^\top x_i+b))$
* Train with full-batch subgradient descent; plot decision boundary and margins

**Requirements**

* **≥ 95% training accuracy**
* Plot: $f(x)=0$ and margins $f(x)=\pm 1$
* Curve: objective vs epochs

**Part B: Dual + Kernelized SVM (Projected Gradient Ascent)**

* Implement kernels: **linear**, **RBF**, **polynomial**
* Dual gradient and **projections**:

  * Box: $0\le \alpha_i\le C$
  * Equality: $\sum_i \alpha_i y_i = 0$
* Recover bias from support vectors
* Visualize boundary and mark SVs

**Requirements**

* **≥ 90% test accuracy**
* Plot: decision boundary + SVs, dual objective curve

**Part C: Dual + Kernel C-SVM via SMO with KKT Checks**

* Two-variable updates with bounds $(L,H)$, curvature $\eta$, and bias updates ($b_1, b_2$ rules)
* Explicit KKT violation metrics and stopping
* Error cache maintenance

**Requirements**

* **≥ 90% test accuracy**
* Diagnostics printed: equality residual, max KKT violation, duality gap
* Plots: boundary + SVs, KKT violation over iterations, dual objective trend

---

### Q4 — Trees and Ensembles (25 pts)

**Part A: Unified CART (Classification & Regression)**
Implement a single tree that supports:

* **Classification**: Gini impurity; majority vote at leaves
* **Regression**: MSE; mean at leaves
* Binary splits, candidate thresholds as midpoints, stopping criteria:

  * `max_depth`, `min_samples_split`, `min_samples_leaf`

**Requirements**

* **Classification:** depth=3 CART **≥ 0.90 test accuracy**, not worse than logistic regression baseline; plot decision boundary
* **Regression:** **test MSE < 0.10**, better than linear baseline; plot regression curve

**Part B: Post-Pruning (Reduced-Error)**

* Build a large tree, then prune bottom-up using a validation set
* Replace subtrees with a leaf if validation score improves (or ties, per policy)

**Requirements**

* **Classification:** after pruning, **train accuracy decreases** while **test accuracy increases**; boundary simpler
* **Regression:** after pruning, **train MSE increases** while **test MSE decreases**; curve smoother
* Plots: before vs after pruning

**Part C: Random Forest (Bagging, Feature Subsampling, OOB, FI)**
Implement RF using your CART:

* Bootstrap samples per tree
* Feature subsampling per split (`max_features`)
* Aggregation: majority vote (clf) or mean (reg)
* **OOB predictions** and **OOB score**
* **Feature Importance**

  * Impurity-based (sum of split gains, sample-weighted)
  * Permutation (OOB-based)

**Requirements**

* **Classification:** RF **test accuracy > single tree**; bar chart (Impurity FI vs Permutation FI)
* **Regression (1D & 2D):** RF **test MSE < single tree**; in 2D, FI should show $x_0$ (sin) dominates, $x_1$ (linear) contributes
* Plots: OOB vs test, decision/regression visuals, FI bars

---


## 🧪 Baselines & Thresholds (Quick Reference)

* **Q2A:** LR MSE < 0.65 (closed-form & GD)
* **Q2B:** Logistic accuracy = 100%
* **Q2C:** SVR MSE < LR and < 0.05
* **Q3A:** Linear SVM train acc ≥ 95%
* **Q3B/C:** Kernel SVM test acc ≥ 90%
* **Q4A:** CART clf depth=3 test acc ≥ 0.90; CART reg MSE < 0.10
* **Q4B:** After pruning, classification test acc ↑ (train acc ↓); regression test MSE ↓ (train MSE ↑)
* **Q4C:** RF > single tree (acc for clf, lower MSE for reg). FI: in 2D reg, $x_0$ dominates, $x_1$ contributes.

---

