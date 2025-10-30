# Machine Learning Assignment 2 — Clustering, Dimensionality Reduction, CNNs, and RNNs

> End-to-end ML systems from scratch: implementing core algorithms, visualize what they learn, and benchmark them on real datasets.

<p align="center">
  <b>Q1</b> Clustering • <b>Q2</b> Dimensionality Reduction • <b>Q3</b> CNNs from NumPy • <b>Q4</b> RNN/LSTM/GRU in PyTorch
</p>

---

## 📦 What’s inside

* **Notebook**: `ML_Assignment2_template.ipynb`
  Contains all four questions with TODO blocks  must fill.

* **Datasets (auto-downloaded)**

  * Synthetic 2D data (generated in code)
  * `sklearn` Digits, OpenML **MNIST**
  * **CIFAR-10** (via `tensorflow.keras.datasets` in Q3; via `torchvision` in Q4)

---

## 🚀 Quick start

> **GPU (Q4 suggested):** In Colab/Studio, set **Runtime → Change runtime type → GPU (e.g., T4)**.

---

### Q1 — Clustering (Foundations & from-scratch K-Means)

* Implement **K-Means** (init, assignment, update, convergence).
* Understand why K-Means fails on **concentric Gaussians** and how **GMM/EM** fixes it via **soft probabilistic assignments** & covariance.
* Know what **σ** does in **RBF affinities** (Spectral Clustering) and **DBSCAN** behavior on noise.
* Metrics: silhouette/NMI/ARI (optional).

### Q2 — Dimensionality Reduction (PCA → ICA → t-SNE + Pseudo-labels)

* Implement **PCA** from scratch: centering, covariance, eigendecomposition, sorting, projection.
* Implement **FastICA** (fixed-point) with whitening + decorrelation.
* Compare **PCA vs t-SNE** for visualization.
* Build a **pseudo-label** pipeline:

  1. PCA(→50) → 2) **K-Means** → 3) **Logistic Regression** trained on pseudo-labels → 4) t-SNE visualization.

**Targets**

* PCA explained-variance within **0.05** of `sklearn.PCA`.
* ICA source recovery corr. **≥ 0.90** (up to permutation/sign).
* Pseudo-label LR test acc **≥ 0.95** and t-SNE silhouette **≥ 0.2**.

### Q3 — CNNs from scratch in NumPy (no DL frameworks)

* Implement low-level components:

  * `im2col/col2im`, **Conv2D**, **MaxPool**, **BatchNorm**, **Dropout**, **Linear**, **ReLU**, **Softmax-CE**, **SGD**.
* Train a CNN on **MNIST** (NumPy only).
* Add **ResNet** residual blocks and compare to a plain CNN on **CIFAR-10**.
* Implement **Depthwise + Pointwise (1×1)** conv (MobileNet-lite); compare params/FLOPs/accuracy.

**Targets**

* MNIST test acc **≥ 0.97**.
* ResNet model test acc **≥ 0.45** on the CIFAR-10 subset and/or faster convergence than plain CNN.
* MobileNet-lite achieves **≥ 75% parameter reduction** with bounded accuracy drop (≤ *reduction − 30%*).

### Q4 — Sequence Models in PyTorch (RNN/LSTM/GRU)

* **RNN** on **Row-wise MNIST** (28 steps × 28 features) + gradient clipping.
* Show RNN limitation on **Row-wise CIFAR-10** (32 steps × 96 features).
* **LSTM** on Row-wise CIFAR-10 → should **outperform RNN**.
* **Failure case**: LSTM on **Pixel-wise MNIST** (784 steps × 1 feature).
* **GRU + gradient clipping** on Pixel-wise MNIST → improves over LSTM.

**Targets**

* RNN (row-wise MNIST) test acc **≥ 0.97**.
* LSTM > RNN on row-wise CIFAR-10.
* GRU on pixel-wise MNIST **significantly higher** than LSTM baseline.

---

## 🧭 workflow

1. **Q1**: Finish K-Means & theory checks → sanity-plot concentric rings.
2. **Q2**: Implement PCA & ICA → verify variance/correlation → run pseudo-label pipeline + t-SNE.
3. **Q3**: Bring up Conv/Pool/BN/Dropout/Linear/Softmax → hit MNIST target → add ResBlock → implement Depthwise/Pointwise & compare params.
4. **Q4**: Switch to PyTorch → RNN MNIST (row-wise) → reproduce failure on CIFAR row-wise → LSTM improvement → GRU + clipping on pixel-wise MNIST.

---

## 📊 Expected plots & artifacts

* Q1: Cluster assignments, decision boundaries; DBSCAN noise points.
* Q2: PCA 2-D scatter, ICA waveforms, t-SNE embedding with thumbnails.
* Q3: Training loss & accuracy curves (MNIST, CIFAR); parameter/FLOP comparison tables.
* Q4: Training loss; test accuracy for RNN/LSTM/GRU; short failure-case notes.

---

## 🛠️ Tips & troubleshooting

* **Numerics**: add small eps (e.g., `1e-12`) to norms/variances; clip logits in CE if needed.
* **BatchNorm (from scratch)**: keep running mean/var for eval; mind broadcasting shapes.
* **ICA**: ensure proper whitening and **symmetric decorrelation** for stable convergence.
* **t-SNE**: try `perplexity ∈ {20, 30, 40}`; subsample if it’s slow.
* **Gradient clipping (Q4)**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`.
