#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, re, os, sys, math, glob, pathlib, warnings
from typing import List, Tuple
import pandas as pd
import numpy as np
from torch.utils.data import Sampler
import random, math

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

# ---------- Training: Supervised-Contrastive on opening heatmaps ----------
# Requires: pip install torch
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ClassBalancedBatchSampler(Sampler):
    """
    每個 batch 取 `classes_per_batch` 個類別、每類 `samples_per_class` 筆。
    以「有放回抽樣」避免類別資料被抽光；同時給定每個 epoch 的批次數，讓 len(dataloader) 有意義。
    """
    def __init__(self, labels, classes_per_batch=32, samples_per_class=8, batches_per_epoch=None):
        self.labels = np.asarray(labels)
        self.idxs_by_cls = {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}
        self.classes = list(self.idxs_by_cls.keys())
        self.cpb = classes_per_batch
        self.spc = samples_per_class
        # 一個 epoch 大約覆蓋一次資料量
        if batches_per_epoch is None:
            denom = max(1, self.cpb * self.spc)
            batches_per_epoch = int(math.ceil(len(self.labels) / denom))
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_cls = random.sample(self.classes, min(self.cpb, len(self.classes)))
            batch = []
            for c in chosen_cls:
                pool = self.idxs_by_cls[c]
                # 有放回抽樣 -> 不會出現 sample larger than population
                take = np.random.choice(pool, size=self.spc, replace=True)
                batch.extend(take.tolist())
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch

def game_vector_from_text(sgf_text, board=19, grid=13, max_moves=60):
    moves = extract_moves_from_game(sgf_text, board=board, max_moves=max_moves)
    return game_style_vector(moves, board=board, grid=grid)  # (2*grid*grid,)

def index_first_k_games_pack(path, keep_games=250, **featkw):
    """Return up to K per-game vectors + player label from a single training file."""
    try:
        txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = pathlib.Path(path).read_text(encoding="latin-1", errors="ignore")
    games = split_games_from_pack(txt)
    vecs = []
    for g in games[:keep_games]:
        v = game_vector_from_text(g, **featkw)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    X = np.stack(vecs, 0).astype(np.float32)
    return X

class TrainSet(Dataset):
    def __init__(self, root, grid=13, max_moves=60, keep_games=250):
        self.X, self.y = [], []
        self.dim = 2*grid*grid
        self.players = sorted(pathlib.Path(root).glob("*.sgf"), key=lambda p: numeric_id_from_stem(p.stem))
        for p in self.players:
            pid = numeric_id_from_stem(p.stem)  # 1..200
            Xp = index_first_k_games_pack(str(p), keep_games=keep_games, board=19, grid=grid, max_moves=max_moves)
            if Xp is None: 
                continue
            self.X.append(Xp)
            self.y.append(np.full((len(Xp),), pid-1, dtype=np.int64))  # labels 0..199
        self.X = np.concatenate(self.X, 0)
        self.y = np.concatenate(self.y, 0)
        # standardize per-dim
        self.mu = self.X.mean(0, keepdims=True); self.sd = self.X.std(0, keepdims=True)+1e-6
        self.X = (self.X - self.mu)/self.sd

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): 
        return self.X[i], self.y[i]

class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim=256, pdrop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(True), nn.BatchNorm1d(512), nn.Dropout(pdrop),
            nn.Linear(512, emb_dim), nn.ReLU(True),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)

def supcon_loss(z, y, T=0.07):
    """
    Supervised-Contrastive loss (Khosla et al. 2020)
    z: [B, D] L2-normalized embeddings
    y: [B] int labels
    """
    # pairwise similarities
    sim = (z @ z.t()) / T                    # [B,B]
    # remove self-comparisons from the denominator via logsumexp mask
    B = sim.size(0)
    eye = torch.eye(B, device=sim.device, dtype=torch.bool)

    # mask of positives (same label), but **exclude self on the diagonal**
    pos_mask = (y.unsqueeze(1) == y.unsqueeze(0)) & (~eye)   # [B,B] bool

    # log-softmax over all non-self examples
    sim_no_self = sim.masked_fill(eye, float('-inf'))        # -inf on diag
    log_prob = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

    # for each anchor, average log-probs over its positives
    pos_count = pos_mask.sum(dim=1).clamp(min=1)             # avoid /0
    loss = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count)
    return loss.mean()

def train_encoder(train_root, save_path, grid=13, max_moves=60, keep_games=250,
                  epochs=10, batch_size=512, lr=1e-3, emb_dim=256, device="cuda" if torch.cuda.is_available() else "cpu"):
    tr = TrainSet(train_root, grid=grid, max_moves=max_moves, keep_games=keep_games)
    sampler = ClassBalancedBatchSampler(tr.y, classes_per_batch=32, samples_per_class=8)
    loader = DataLoader(tr, batch_sampler=sampler, num_workers=2, pin_memory=torch.cuda.is_available())
    enc = Encoder(in_dim=tr.dim, emb_dim=emb_dim).to(device)
    opt = optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[Train] samples={len(tr)} dim={tr.dim} emb={emb_dim} device={device}")
    for ep in range(1, epochs+1):
        enc.train(); tot=0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            zb = enc(xb)
            loss = supcon_loss(zb, yb, T=0.07)
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 3.0)
            opt.step()
            tot += loss.item()
        print(f"[Train] epoch {ep:02d} | loss {tot/len(loader):.4f}")
    ckpt = {"state_dict": enc.state_dict(), "mu": tr.mu, "sd": tr.sd, "dim": tr.dim, "grid": grid, "max_moves": max_moves, "emb_dim": emb_dim}
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[Train] saved -> {save_path}")

# ---- integrate with inference: if a checkpoint is provided, use it to embed files ----
def load_encoder(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) 
    enc = Encoder(in_dim=ckpt["dim"], emb_dim=ckpt["emb_dim"]).to(device)
    enc.load_state_dict(ckpt["state_dict"]); enc.eval()
    return enc, ckpt

def file_embedding_trained(path, enc, ckpt, device, avg_games=120):
    """Average per-game embeddings (using trained encoder) for a player file."""
    # Read and split once
    try:
        txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = pathlib.Path(path).read_text(encoding="latin-1", errors="ignore")
    games = split_games_from_pack(txt)
    if not games: games = [txt]
    # Subsample to speed up
    if len(games) > avg_games:
        idx = np.random.default_rng(123).choice(len(games), size=avg_games, replace=False)
        games = [games[i] for i in idx]
    X = []
    for g in games:
        v = game_vector_from_text(g, board=19, grid=ckpt["grid"], max_moves=ckpt["max_moves"])
        X.append(v)
    X = np.stack(X,0).astype(np.float32)
    # standardize like training
    X = (X - ckpt["mu"]) / ckpt["sd"]
    with torch.no_grad():
        z = enc(torch.from_numpy(X).to(device))
        z = nn.functional.normalize(z, dim=-1)
        z = z.mean(0, keepdim=False)
        z = nn.functional.normalize(z, dim=0)
    return z.cpu().numpy()

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
    # Detect train/infer mode
    if "--train" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--train", action="store_true")
        ap.add_argument("--data_root", type=str, required=True)          # root that contains train_set/
        ap.add_argument("--save", type=str, default="models/enc.pt")
        ap.add_argument("--grid", type=int, default=13)
        ap.add_argument("--max_moves", type=int, default=60)
        ap.add_argument("--keep_games", type=int, default=250)           # per train file indexed
        ap.add_argument("--epochs", type=int, default=10)
        ap.add_argument("--batch_size", type=int, default=512)
        ap.add_argument("--emb_dim", type=int, default=256)
        ap.add_argument("--lr", type=float, default=1e-3)
        args = ap.parse_args()

        train_root = pathlib.Path(args.data_root) / "train_set"
        train_encoder(str(train_root), args.save, grid=args.grid, max_moves=args.max_moves,
                      keep_games=args.keep_games, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, emb_dim=args.emb_dim)
        sys.exit(0)

    # Normal inference path, but check for --load
    # Reuse your original argparse in main(), or:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out", type=str, default="submission.csv")
    ap.add_argument("--board", type=int, default=19)
    ap.add_argument("--grid", type=int, default=19)
    ap.add_argument("--max_moves", type=int, default=50)
    ap.add_argument("--load", type=str, default="", help="path to trained encoder .pt (optional)")
    args = ap.parse_args()

    query_dir = pathlib.Path(args.data_root) / "test_set" / "query_set"
    cand_dir  = pathlib.Path(args.data_root) / "test_set" / "cand_set"

    # Build candidate & query embeddings
    if args.load and pathlib.Path(args.load).exists():
        enc, ckpt = load_encoder(args.load)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Infer] using trained encoder: {args.load}  device={device}")
        # candidates
        cand_files = sorted(pathlib.Path(cand_dir).glob("*.sgf"), key=lambda p: numeric_id_from_stem(p.stem))
        cand_ids, cand_embs = [], []
        for p in cand_files:
            cid = numeric_id_from_stem(p.stem)
            z = file_embedding_trained(str(p), enc, ckpt, device)
            cand_ids.append(cid); cand_embs.append(z)
        cand_ids = np.array(cand_ids, dtype=np.int32)
        cand_embs = np.stack(cand_embs,0)
        cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True)+1e-12)

        # queries
        query_files = sorted(pathlib.Path(query_dir).glob("*.sgf"), key=lambda p: numeric_id_from_stem(p.stem))
        recs = []
        for qp in query_files:
            qid = numeric_id_from_stem(qp.stem)
            qz = file_embedding_trained(str(qp), enc, ckpt, device)
            sims = cand_embs @ qz
            best = int(cand_ids[np.argmax(sims)])
            recs.append({"id": qid, "label": best})
        pd.DataFrame(recs).sort_values("id").to_csv(args.out, index=False)
        print(f"[Infer] wrote {args.out}")
    else:
        # Fallback to your original non-trained baseline main()
        main()
