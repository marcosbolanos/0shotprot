#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES

DEFAULT_TASKS = ["E4B", "AMIE", "LGK", "Pab1", "TEM", "UBE2I", "GFP", "AAV"]
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ESM-2 masked-marginal zero-shot alignment on ProSpero validation sets.")
    p.add_argument("--output_dir", default="outputs/esm2_650m_zero_shot_alignment_20260603/predictive_scores")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--split", choices=["valid", "train", "all"], default="valid")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="esm2_t33_650M_UR50D", choices=["esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    return p.parse_args()


def normalize_sequence(seq):
    if isinstance(seq, str):
        return seq
    return "".join(map(str, seq))


def collapsed_split(task, split):
    dataset = RegressionDataset(task)
    if split == "valid":
        seqs, scores = dataset.valid, dataset.valid_scores
    elif split == "train":
        seqs, scores = dataset.train, dataset.train_scores
    else:
        seqs = np.concatenate((dataset.train, dataset.valid), axis=0)
        scores = np.concatenate((dataset.train_scores, dataset.valid_scores), axis=0)
    collapsed = {}
    for seq, score in zip(seqs, scores):
        seq = normalize_sequence(seq)
        collapsed.setdefault(seq, []).append(float(score))
    return [(seq, float(np.mean(vals))) for seq, vals in collapsed.items()]


def load_completed(path):
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed[row["sequence"]] = row
    return completed


def make_writer(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def mutation_positions(seq, wt):
    if len(seq) != len(wt):
        raise ValueError(f"Sequence length {len(seq)} does not match WT length {len(wt)}")
    return [idx for idx, (aa, wt_aa) in enumerate(zip(seq, wt)) if aa != wt_aa]


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def top_recall(y_true, y_pred, fraction=0.1):
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(n * fraction)))
    true_top = set(np.argsort(y_true)[::-1][:k])
    pred_top = set(np.argsort(y_pred)[::-1][:k])
    return len(true_top & pred_top) / k


def ndcg(y_true, y_pred, fraction=0.1):
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(n * fraction)))
    gains = np.asarray(y_true, dtype=float)
    gains = gains - np.min(gains)
    pred_order = np.argsort(y_pred)[::-1][:k]
    ideal_order = np.argsort(gains)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[pred_order] * discounts))
    idcg = float(np.sum(gains[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def summarize(task, rows, out_dir, model_name, seconds=None):
    if not rows:
        return None
    y = np.array([float(r["fitness"]) for r in rows], dtype=float)
    pred = np.array([float(r["masked_marginals"]) for r in rows], dtype=float)
    summary = {
        "task": task,
        "model": model_name,
        "n_sequences": int(len(rows)),
        "fitness_min": float(np.min(y)),
        "fitness_median": float(np.median(y)),
        "fitness_max": float(np.max(y)),
        "seconds": seconds,
        "masked_marginals": {
            "spearman": safe_corr(rankdata(pred), rankdata(y)),
            "pearson": safe_corr(pred, y),
            "ndcg_top10pct": ndcg(y, pred, fraction=0.1),
            "top10pct_recall": top_recall(y, pred, fraction=0.1),
            "top100_overlap": len(set(np.argsort(y)[::-1][: min(100, len(y))]) & set(np.argsort(pred)[::-1][: min(100, len(y))])),
        },
    }
    with (out_dir / f"{task}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


class Esm2Scorer:
    def __init__(self, model, alphabet, device, batch_size):
        self.model = model
        self.alphabet = alphabet
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.batch_converter = alphabet.get_batch_converter()
        self.aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA20}

    @torch.inference_mode()
    def masked_marginal_position_log_probs(self, wt):
        data = [(f"pos_{i}", wt) for i in range(len(wt))]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        for idx in range(len(wt)):
            tokens[idx, idx + 1] = self.alphabet.mask_idx
        all_log_probs = []
        aa_cols = torch.tensor([self.aa_to_idx[aa] for aa in AA20], device=self.device)
        for start in range(0, len(wt), self.batch_size):
            chunk = tokens[start : start + self.batch_size]
            out = self.model(chunk, repr_layers=[], return_contacts=False)
            logits = out["logits"]
            rows = torch.arange(chunk.shape[0], device=self.device)
            positions = torch.arange(start, min(start + self.batch_size, len(wt)), device=self.device) + 1
            selected = logits[rows, positions][:, aa_cols]
            all_log_probs.append(F.log_softmax(selected, dim=1).detach().cpu())
        return torch.cat(all_log_probs, dim=0).numpy()

    def score_masked_marginals(self, seq, wt, wt_position_log_probs):
        score = 0.0
        aa_to_col = {aa: i for i, aa in enumerate(AA20)}
        for idx in mutation_positions(seq, wt):
            mut_col = aa_to_col.get(seq[idx])
            wt_col = aa_to_col.get(wt[idx])
            if mut_col is None or wt_col is None:
                continue
            score += float(wt_position_log_probs[idx, mut_col] - wt_position_log_probs[idx, wt_col])
        return score


def score_task(args, task, scorer):
    out_dir = Path(args.output_dir)
    rows_path = out_dir / f"{task}.csv"
    fieldnames = ["task", "split", "sequence", "fitness", "n_mutations", "masked_marginals"]
    completed = load_completed(rows_path)
    split_rows = collapsed_split(task, args.split)
    if args.max_sequences is not None:
        split_rows = split_rows[: args.max_sequences]
    pending = [(seq, fitness) for seq, fitness in split_rows if seq not in completed]
    wt = WT_SEQUENCES[task]
    t0 = time.perf_counter()
    wt_log_probs = scorer.masked_marginal_position_log_probs(wt) if pending else None
    handle, writer = make_writer(rows_path, fieldnames)
    try:
        for idx, (seq, fitness) in enumerate(pending, start=1):
            row = {
                "task": task,
                "split": args.split,
                "sequence": seq,
                "fitness": f"{fitness:.10g}",
                "n_mutations": len(mutation_positions(seq, wt)),
                "masked_marginals": f"{scorer.score_masked_marginals(seq, wt, wt_log_probs):.10g}",
            }
            writer.writerow(row)
            if idx % 50 == 0:
                handle.flush()
        handle.flush()
    finally:
        handle.close()
    seconds = time.perf_counter() - t0
    all_rows = list(load_completed(rows_path).values())
    summary = summarize(task, all_rows, out_dir, args.model, seconds=seconds)
    if summary:
        print(json.dumps(summary, sort_keys=True), flush=True)


def load_model(name, device):
    import esm
    loader = getattr(esm.pretrained, name)
    model, alphabet = loader()
    model.eval().to(device)
    return model, alphabet


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, alphabet = load_model(args.model, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"event": "model_loaded", "model": args.model, "seconds": time.perf_counter() - t0, "params": sum(p.numel() for p in model.parameters())}), flush=True)
    scorer = Esm2Scorer(model, alphabet, args.device, args.batch_size)
    for task in args.tasks:
        print(f"Scoring {task}", flush=True)
        score_task(args, task, scorer)


if __name__ == "__main__":
    main()
