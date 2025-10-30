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

---

## 5) Expected Results

With the plain baseline command above, I obtained:

```
Public LB ≈ 0.8333
```

Small variations come from heat-map grid, move count, and file parsing differences.


