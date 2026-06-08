import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES


DEFAULT_TASKS = ["E4B", "AMIE", "LGK", "Pab1", "TEM", "UBE2I", "GFP", "AAV"]
AA_COUNT = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EvoDiff zero-shot alignment on ProSpero validation sets. "
            "Scores are relative to the WT sequence and duplicates are collapsed."
        )
    )
    parser.add_argument("--output_dir", default="outputs/evodiff_zero_shot_alignment_20260602/predictive_scores")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split", choices=["valid", "train", "all"], default="valid")
    parser.add_argument("--methods", nargs="+", choices=["masked_marginals", "generation_path"], default=["masked_marginals", "generation_path"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    rows = [(seq, float(np.mean(vals))) for seq, vals in collapsed.items()]
    return rows


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
        raise ValueError(f"Sequence length {len(seq)} does not match WT length {len(wt)}.")
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


def summarize(task, rows, out_dir):
    if not rows:
        return
    y = np.array([float(r["fitness"]) for r in rows], dtype=float)
    summary = {
        "task": task,
        "n_sequences": int(len(rows)),
        "fitness_min": float(np.min(y)),
        "fitness_median": float(np.median(y)),
        "fitness_max": float(np.max(y)),
    }
    for method in ("masked_marginals", "generation_path"):
        if method not in rows[0] or rows[0][method] == "":
            continue
        pred = np.array([float(r[method]) for r in rows], dtype=float)
        summary[method] = {
            "spearman": safe_corr(rankdata(pred), rankdata(y)),
            "pearson": safe_corr(pred, y),
            "ndcg_top10pct": ndcg(y, pred, fraction=0.1),
            "top10pct_recall": top_recall(y, pred, fraction=0.1),
            "top100_overlap": len(set(np.argsort(y)[::-1][: min(100, len(y))]) & set(np.argsort(pred)[::-1][: min(100, len(y))])),
        }
    with (out_dir / f"{task}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


class EvoDiffScorer:
    def __init__(self, model, tokenizer, device, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def aa_token(self, aa):
        tokenized = self.tokenizer.tokenize([aa])
        return int(tokenized[0])

    def tokenized_wt(self, wt):
        return np.asarray(self.tokenizer.tokenize([wt]), dtype=np.int64)

    @torch.no_grad()
    def masked_marginal_position_log_probs(self, wt):
        tokenized = self.tokenized_wt(wt)
        rows = np.tile(tokenized, (len(wt), 1))
        for idx in range(len(wt)):
            rows[idx, idx] = self.tokenizer.mask_id

        all_log_probs = []
        for start in range(0, len(rows), self.batch_size):
            chunk = torch.tensor(rows[start : start + self.batch_size], device=self.device)
            timestep = torch.zeros(chunk.shape[0], dtype=torch.long, device=self.device)
            logits = self.model(chunk, timestep)
            positions = torch.arange(start, min(start + self.batch_size, len(rows)), device=self.device)
            batch_rows = torch.arange(chunk.shape[0], device=self.device)
            selected = logits[batch_rows, positions, :AA_COUNT]
            all_log_probs.append(F.log_softmax(selected, dim=1).cpu())
        return torch.cat(all_log_probs, dim=0).numpy()

    def score_masked_marginals(self, seq, wt, wt_position_log_probs):
        score = 0.0
        for idx in mutation_positions(seq, wt):
            mut_token = self.aa_token(seq[idx])
            wt_token = self.aa_token(wt[idx])
            score += float(wt_position_log_probs[idx, mut_token] - wt_position_log_probs[idx, wt_token])
        return score

    @torch.no_grad()
    def score_generation_path(self, items, wt):
        scores = {seq: 0.0 for seq in items}
        active = []
        tokenized_wt = self.tokenized_wt(wt)
        for seq in items:
            muts = mutation_positions(seq, wt)
            if muts:
                active.append(
                    {
                        "seq": seq,
                        "muts": muts,
                        "step": 0,
                        "tokens": tokenized_wt.copy(),
                        "score": 0.0,
                    }
                )

        while active:
            next_active = []
            for start in range(0, len(active), self.batch_size):
                chunk_items = active[start : start + self.batch_size]
                rows = np.stack([item["tokens"] for item in chunk_items], axis=0)
                positions = np.array([item["muts"][item["step"]] for item in chunk_items], dtype=np.int64)
                chunk = torch.tensor(rows, device=self.device)
                timestep = torch.zeros(chunk.shape[0], dtype=torch.long, device=self.device)
                logits = self.model(chunk, timestep)
                batch_rows = torch.arange(chunk.shape[0], device=self.device)
                log_probs = F.log_softmax(logits[batch_rows, torch.tensor(positions, device=self.device), :AA_COUNT], dim=1).cpu().numpy()

                for row_idx, item in enumerate(chunk_items):
                    pos = int(positions[row_idx])
                    mut_token = self.aa_token(item["seq"][pos])
                    wt_token = self.aa_token(wt[pos])
                    item["score"] += float(log_probs[row_idx, mut_token] - log_probs[row_idx, wt_token])
                    item["tokens"][pos] = mut_token
                    item["step"] += 1
                    if item["step"] < len(item["muts"]):
                        next_active.append(item)
                    else:
                        scores[item["seq"]] = item["score"]
            active = next_active
        return scores


def score_task(args, task, scorer):
    out_dir = Path(args.output_dir)
    rows_path = out_dir / f"{task}.csv"
    fieldnames = [
        "task",
        "split",
        "sequence",
        "fitness",
        "n_mutations",
        "masked_marginals",
        "generation_path",
    ]
    completed = load_completed(rows_path)
    split_rows = collapsed_split(task, args.split)
    if args.max_sequences is not None:
        split_rows = split_rows[: args.max_sequences]
    pending = [(seq, fitness) for seq, fitness in split_rows if seq not in completed]
    wt = WT_SEQUENCES[task]

    handle, writer = make_writer(rows_path, fieldnames)
    try:
        wt_log_probs = None
        if "masked_marginals" in args.methods and pending:
            wt_log_probs = scorer.masked_marginal_position_log_probs(wt)

        generation_scores = {}
        if "generation_path" in args.methods and pending:
            generation_scores = scorer.score_generation_path([seq for seq, _ in pending], wt)

        for idx, (seq, fitness) in enumerate(pending, start=1):
            row = {
                "task": task,
                "split": args.split,
                "sequence": seq,
                "fitness": f"{fitness:.10g}",
                "n_mutations": len(mutation_positions(seq, wt)),
                "masked_marginals": "",
                "generation_path": "",
            }
            if wt_log_probs is not None:
                row["masked_marginals"] = f"{scorer.score_masked_marginals(seq, wt, wt_log_probs):.10g}"
            if generation_scores:
                row["generation_path"] = f"{generation_scores[seq]:.10g}"
            writer.writerow(row)
            if idx % 50 == 0:
                handle.flush()
        handle.flush()
    finally:
        handle.close()

    all_rows = list(load_completed(rows_path).values())
    summary = summarize(task, all_rows, out_dir)
    if summary:
        print(json.dumps(summary, sort_keys=True))


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

    model, _, tokenizer, _ = OA_DM_38M()
    model = model.to(args.device)
    model.eval()

    scorer = EvoDiffScorer(model, tokenizer, args.device, args.batch_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        print(f"Scoring {task}")
        score_task(args, task, scorer)


if __name__ == "__main__":
    main()
