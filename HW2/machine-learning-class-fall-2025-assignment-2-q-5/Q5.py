#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5 — Who play this game?  (baseline submission generator)

Pure-Python SGF parser + simple style features (opening heatmaps) + cosine NN.
No external dependencies beyond NumPy & Pandas.

Usage
-----
python Q5.py --data_root /path/to/HW2 \
             --out submission.csv \
             --max_moves 50 --grid 19

Directory layout expected
-------------------------
data_root/
  train_set/           # not used by this baseline
  test_set/
    query_set/        # 600 files: player??? or 1.sgf ... 600.sgf
    cand_set/         # 600 files: player??? or 1.sgf ... 600.sgf
"""
from __future__ import annotations
import argparse, re, os, sys, math, glob, pathlib, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd

# ---------- SGF parsing helpers ----------
SGF_MOVE_RE = re.compile(r";\s*([BW])\s*\[\s*([a-z]{0,2})\s*\]", re.IGNORECASE)
SGF_GAME_RE = re.compile(r"\(\s*;(?:(?!\)\s*\().|\n)*?\)", re.DOTALL)  # non-greedy SGF root .. ')'

def _sgf_coord_to_xy(coord: str, board: int = 19) -> Tuple[int, int]:
    """SGF coordinates are 'aa'.. ; empty '' or 'tt' may mean pass (we ignore)."""
    if len(coord) != 2:  # pass or empty
        return (-1, -1)
    x = ord(coord[0]) - ord('a')
    y = ord(coord[1]) - ord('a')
    if 0 <= x < board and 0 <= y < board:
        return (x, y)
    return (-1, -1)

def extract_moves_from_game(sgf_text: str, board: int = 19, max_moves: int = 50) -> List[Tuple[str,int,int]]:
    """Return first max_moves as (color, x, y). Ignore passes/out-of-board."""
    out = []
    for m in SGF_MOVE_RE.finditer(sgf_text):
        c = m.group(1).upper()
        x, y = _sgf_coord_to_xy(m.group(2).lower(), board)
        if x >= 0:
            out.append((c, x, y))
            if len(out) >= max_moves:
                break
    return out

def split_games_from_pack(s: str) -> List[str]:
    # Most packs are concatenated SGFs: "(;...)(;...)(;...)"
    return SGF_GAME_RE.findall(s)

# ---------- Feature extraction ----------
def game_style_vector(moves: List[Tuple[str,int,int]],
                      board: int = 19,
                      grid: int = 19) -> np.ndarray:
    """
    Opening heatmap for first K moves, separated by color (B/W).
    If grid < board, we downsample each coordinate to a coarse grid (e.g., 13 or 9).
    Returns shape (2*grid*grid,).
    """
    if grid <= 0: grid = board
    scale = (grid-1)/(board-1) if board > 1 else 1.0

    hb = np.zeros((grid, grid), dtype=np.float32)
    hw = np.zeros((grid, grid), dtype=np.float32)

    for c,x,y in moves:
        gx = int(round(x*scale))
        gy = int(round(y*scale))
        if c == 'B':
            hb[gy, gx] += 1.0
        else:
            hw[gy, gx] += 1.0

    # Normalize each channel by total moves for stability
    if hb.sum() > 0: hb /= hb.sum()
    if hw.sum() > 0: hw /= hw.sum()
    return np.concatenate([hb.flatten(), hw.flatten()], axis=0)

def file_embedding(path: str,
                   board: int = 19,
                   grid: int = 19,
                   max_moves: int = 50) -> np.ndarray:
    """
    Average the per-game opening vectors to get a per-player-file embedding.
    """
    try:
        txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = pathlib.Path(path).read_text(encoding="latin-1", errors="ignore")

    games = split_games_from_pack(txt)
    if not games:
        # Fallback: treat the whole file as one game
        games = [txt]

    vecs = []
    for g in games:
        moves = extract_moves_from_game(g, board=board, max_moves=max_moves)
        if moves:
            vecs.append(game_style_vector(moves, board=board, grid=grid))

    if not vecs:
        # Empty? return zeros to keep pipeline running
        return np.zeros((2*grid*grid,), dtype=np.float32)

    V = np.stack(vecs, axis=0)
    V = V.mean(axis=0)
    # L2-normalize for cosine space
    n = np.linalg.norm(V) + 1e-12
    return (V / n).astype(np.float32)

# ---------- Utility: ID parsing & I/O ----------
def numeric_id_from_stem(stem: str) -> int:
    """
    Turn 'player001' or '001' or '1' into integer 1.
    """
    m = re.findall(r"\d+", stem)
    if not m:
        raise ValueError(f"Cannot parse numeric id from stem='{stem}'")
    return int(m[-1].lstrip("0") or "0")

def list_sgf(dir_path: str) -> List[pathlib.Path]:
    ps = sorted(pathlib.Path(dir_path).glob("*.sgf"))
    if not ps:
        raise FileNotFoundError(f"No .sgf under {dir_path}")
    return ps

# ---------- Main inference ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root folder that contains test_set/")
    ap.add_argument("--out", type=str, default="submission.csv")
    ap.add_argument("--board", type=int, default=19)
    ap.add_argument("--grid", type=int, default=19, help="downsample grid for heatmap (e.g., 13/9/19)")
    ap.add_argument("--max_moves", type=int, default=50, help="moves per game to use for opening style")
    args = ap.parse_args()

    query_dir = pathlib.Path(args.data_root) / "test_set" / "query_set"
    cand_dir  = pathlib.Path(args.data_root) / "test_set" / "cand_set"

    print(f"[Q5] Loading candidates from: {cand_dir}")
    cand_files = list_sgf(cand_dir)

    cand_ids, cand_embs = [], []
    for p in cand_files:
        cid = numeric_id_from_stem(p.stem)
        emb = file_embedding(str(p), board=args.board, grid=args.grid, max_moves=args.max_moves)
        cand_ids.append(cid); cand_embs.append(emb)
    cand_ids  = np.array(cand_ids, dtype=np.int32)
    cand_embs = np.stack(cand_embs, axis=0)        # [C, D]
    # Normalize rows (defensive)
    cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True)+1e-12)

    print(f"[Q5] Loading queries from: {query_dir}")
    query_files = list_sgf(query_dir)

    # We will output rows sorted by numeric id ascending, to match Kaggle sample.
    query_files_sorted = sorted(query_files, key=lambda p: numeric_id_from_stem(p.stem))

    records = []
    for qpath in query_files_sorted:
        qid = numeric_id_from_stem(qpath.stem)
        qemb = file_embedding(str(qpath), board=args.board, grid=args.grid, max_moves=args.max_moves)
        qemb = qemb / (np.linalg.norm(qemb)+1e-12)  # [D]

        # Cosine similarity to all candidates
        sims = cand_embs @ qemb  # [C]
        best = int(cand_ids[np.argmax(sims)])
        records.append({"id": qid, "label": best})

    df = pd.DataFrame(records).sort_values("id")
    df.to_csv(args.out, index=False)
    print(f"[Q5] Wrote {args.out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
