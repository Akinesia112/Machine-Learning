# Q5 — Who Play This Game? (Go Player Identification)

This folder contains my solution for **Question 5** of ML Assignment 2.
It predicts **which player** (in the `test_set/cand_set`) played the games in each file from `test_set/query_set`, and writes a `submission.csv` for Kaggle.

The default pipeline is a **training-free baseline**:

* Parse SGF packs.
* Build **opening move heat-map embeddings** (Black/White channels; first K moves).
* Average per game → per file.
* **Cosine nearest neighbour** over candidate embeddings.
* Output `submission.csv` (`id,label`).

It already reaches ~**0.83** on the public leaderboard with:

```
python Q5.py --data_root <HW2 root> --out submission.csv --max_moves 50 --grid 19
```

If you also include a training script (see **Optional Training** below), the README explains how to run it.
Either way, `Q5.py` **always** generates `submission.csv` by itself as required by the homework rules.

---

## Directory Layout

```
machine-learning-class-fall-2025-assignment-2-q-5/
├── Q5.py                     # Required: generates submission.csv
├── train.py                  # training
├── Environment/
│   └── requirements.txt      # Minimal Python deps (see below)
├── train_set/                # Provided by TA (not used by baseline)
├── test_set/
│   ├── cand_set/*.sgf        # 600 candidate players
│   └── query_set/*.sgf       # 600 query players
└── (optional training files)
    └── models/               # saved weights (.npz/.pt)
```

> The **baseline** only needs `numpy` and `pandas`.
> The **optional** training examples below still use plain Python/Numpy (no GPU).

---

## 1) Environment Setup

### Option A — Minimal (works for baseline)

```bash
# (Recommended) create a clean environment
conda create -n mlhw2q5 python=3.10 -y
conda activate mlhw2q5

# install minimal deps
pip install -r Environment/requirements.txt
# If you don't want to use the file, this is all that's inside:
#   numpy
#   pandas
```

---

## 2) Quick Start — Generate `submission.csv`

**Windows (CMD/PowerShell)**

```powershell
python Q5.py --data_root "./HW2/machine-learning-class-fall-2025-assignment-2-q-5" `
             --out submission.csv `
             --max_moves 50 --grid 19
```

After running, you should see:

```
[Q5] Wrote submission.csv with 600 rows.
```

Upload `submission.csv` to the Kaggle competition page.

### CLI Arguments

* `--data_root` : path containing `test_set/` (and optionally `train_set/`).
* `--out`       : output CSV path (default: `submission.csv`).
* `--board`     : board size (default 19).
* `--grid`      : heat-map downsample grid (19/13/9). Smaller adds invariance.
* `--max_moves` : number of opening moves to use per game (default 50).

**Tips**

* Try `--grid 13 --max_moves 60` (often similar or slightly better than 19×19).
* This baseline ignores `train_set/` by design; it is a prototype/metric method.

---

## 3) Method Summary (Baseline)

1. **Parse SGF**: split concatenated games `(;...)(;...)` and read moves.
2. **Opening heat-maps**: first *K* moves → two channels (Black/White), normalized.
3. **File embedding**: average game vectors, then **L2-normalize**.
4. **Match**: cosine similarity with all candidate embeddings → **argmax**.
5. **Write CSV**: sorted by numeric id in `query_set` stem (`id,label`).

This is a **few-shot** style prototype: one embedding per candidate.


## 4)What the Code Trains

* **Per-game feature**: opening move **heatmap** (Black/White channels) from the **first K moves**, downsampled to a `grid×grid` histogram → vector of size `2*grid*grid`.
* **Dataset**: from `train_set`, index up to `--keep_games` games **per file (player)** into per-game vectors.
* **Standardization**: save `mu`/`sd` of features in the checkpoint.
* **Encoder**: small MLP → L2-normalized embedding.
* **Loss**: **Supervised Contrastive** (same player = positives).
* **Batching**: class-balanced sampler (many players × few games per player per batch).
* **Stability**: AdamW + grad-clip (3.0).
* **Checkpoint**: `models/enc.pt` contains weights + feature stats & config.

---

## 5) Qick Start

### A) (Baseline, training-free) — already gives ~0.83 with your command

```bash
python train.py --data_root "D:/NTU Courses/ML/HW2/machine-learning-class-fall-2025-assignment-2-q-5" \
             --out submission.csv --max_moves 50 --grid 19
```

### B) Train the encoder (optional; usually improves a few points)

```bash
python train.py --train \
  --data_root "D:/NTU Courses/ML/HW2/machine-learning-class-fall-2025-assignment-2-q-5" \
  --save models/enc.pt \
  --grid 13 --max_moves 60 --keep_games 250 \
  --epochs 10 --batch_size 512 --emb_dim 256 --lr 1e-3
```

**What the flags mean**

* `--grid`: heatmap resolution for features used in training (e.g., 13 or 19).
* `--max_moves`: number of opening moves per game to count (e.g., 50–80).
* `--keep_games`: **per train file** games indexed (speed/memory knob).
* `--emb_dim`: encoder embedding dimension.
* `--epochs`, `--batch_size`, `--lr`: usual training knobs.

> GPU is recommended but not required. Reduce `--batch_size` if you hit OOM.

### C) Inference with the trained encoder

```bash
python Q5.py \
  --data_root "D:/NTU Courses/ML/HW2/machine-learning-class-fall-2025-assignment-2-q-5" \
  --out submission.csv \
  --load models/enc.pt
```

* When `--load` is given, the script:

  1. extracts per-game heatmaps with the **same** `grid/max_moves` stored in the checkpoint,
  2. standardizes by saved `mu`/`sd`,
  3. encodes games → **averages per file** → cosine **nearest neighbor** over `cand_set`,
  4. writes **`submission.csv`** (`id,label`, sorted by id).


---

## 6) Expected Results

With the plain baseline command above, I obtained:

```
Public LB ≈ 0.8333
```

Small variations come from heat-map grid, move count, and file parsing differences.


* **Training**: `models/enc.pt` (weights + feature stats).
* **Inference**: `submission.csv` with 600 rows:
