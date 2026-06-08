#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES


DEFAULT_TASKS = ["E4B", "AMIE", "LGK", "Pab1", "TEM", "UBE2I", "GFP", "AAV"]
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class StructureMapping:
    protein_gym_name: str
    offset: int
    covered_length: int
    identity: float
    note: str = ""


TASK_MAPPINGS = {
    "AAV": StructureMapping("CAPSD_AAV2S_Sinai_2021", 450, 90, 1.0),
    "GFP": StructureMapping("GFP_AEQVI_Sarkisyan_2016", 0, 238, 1.0),
    "AMIE": StructureMapping("AMIE_PSEAE_Wrenbeck_2017", 0, 341, 0.9971),
    "TEM": StructureMapping("BLAT_ECOLX_Firnberg_2014", 0, 286, 0.9965),
    "UBE2I": StructureMapping("UBC9_HUMAN_Weile_2017", 0, 159, 0.9937),
    "Pab1": StructureMapping("PABP_YEAST_Melamed_2013", 125, 75, 0.9733),
    "E4B": StructureMapping("UBE4B_MOUSE_Starita_2013", 1071, 102, 0.9510),
    "LGK": StructureMapping(
        "LGK_LIPST_Klesmith_2015",
        0,
        439,
        0.9732,
        "ProSpero LGK has 8 C-terminal residues without ProteinGym structure tokens; those positions are ignored.",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ProSST ProteinGym-style zero-shot alignment on ProSpero datasets.")
    p.add_argument("--output_dir", default="outputs/prosst_zero_shot_alignment_20260603/predictive_scores")
    p.add_argument("--proteingym_dir", default="outputs/prosst_zero_shot_alignment_20260603/proteingym_benchmark")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--split", choices=["valid", "train", "all"], default="valid")
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    p.add_argument("--structure_vocab_size", default="2048")
    p.add_argument("--allow_low_identity", action="store_true", help="Allow mappings below 95 percent sequence identity.")
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
        collapsed.setdefault(normalize_sequence(seq), []).append(float(score))
    return [(seq, float(np.mean(vals))) for seq, vals in collapsed.items()]


def load_completed(path):
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["sequence"]: row for row in csv.DictReader(handle)}


def make_writer(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def read_fasta_sequence(path):
    seq_lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                seq_lines.append(line)
    return "".join(seq_lines)


def read_structure_tokens(path):
    raw = read_fasta_sequence(path)
    return [int(tok) for tok in raw.split(",") if tok]


def tokenize_structure_tokens(tokens, device):
    shifted = [token + 3 for token in tokens]
    return torch.tensor([[1, *shifted, 2]], dtype=torch.long, device=device)


def mutation_positions(seq, wt, covered_length):
    if len(seq) != len(wt):
        raise ValueError(f"Sequence length {len(seq)} does not match WT length {len(wt)}")
    return [idx for idx, (aa, wt_aa) in enumerate(zip(seq[:covered_length], wt[:covered_length])) if aa != wt_aa]


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


def summarize(task, rows, out_dir, model_name, mapping, seconds=None):
    if not rows:
        return None
    y = np.array([float(r["fitness"]) for r in rows], dtype=float)
    pred = np.array([float(r["prosst_log_odds"]) for r in rows], dtype=float)
    summary = {
        "task": task,
        "model": model_name,
        "n_sequences": int(len(rows)),
        "fitness_min": float(np.min(y)),
        "fitness_median": float(np.median(y)),
        "fitness_max": float(np.max(y)),
        "seconds": seconds,
        "mapping": {
            "protein_gym_name": mapping.protein_gym_name,
            "offset": mapping.offset,
            "covered_length": mapping.covered_length,
            "identity": mapping.identity,
            "note": mapping.note,
        },
        "prosst_log_odds": {
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


class ProSSTScorer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.vocab = tokenizer.get_vocab()

    @torch.inference_mode()
    def wt_position_log_probs(self, wt, structure_tokens):
        tokenized = self.tokenizer([wt], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        ss_input_ids = tokenize_structure_tokens(structure_tokens, self.device)
        if ss_input_ids.shape != input_ids.shape:
            raise ValueError(f"AA tokens shape {tuple(input_ids.shape)} != structure tokens shape {tuple(ss_input_ids.shape)}")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, ss_input_ids=ss_input_ids)
        return F.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)[0].detach().cpu().numpy()

    def score_log_odds(self, seq, wt, log_probs, covered_length):
        score = 0.0
        scored = 0
        skipped = 0
        for idx in mutation_positions(seq, wt, covered_length):
            mut_aa = seq[idx]
            wt_aa = wt[idx]
            if mut_aa not in AA20 or wt_aa not in AA20:
                skipped += 1
                continue
            score += float(log_probs[idx, self.vocab[mut_aa]] - log_probs[idx, self.vocab[wt_aa]])
            scored += 1
        ignored = sum(1 for aa, wt_aa in zip(seq[covered_length:], wt[covered_length:]) if aa != wt_aa)
        return score, scored, skipped, ignored


def load_mapping_assets(args, task):
    mapping = TASK_MAPPINGS.get(task)
    if mapping is None:
        raise ValueError(f"No ProSST structure mapping configured for task {task}")
    if mapping.identity < 0.95 and not args.allow_low_identity:
        raise ValueError(f"Task {task} mapping identity {mapping.identity:.3f} is below 0.95; pass --allow_low_identity to use it.")

    base = Path(args.proteingym_dir)
    residue_path = base / "residue_sequence" / f"{mapping.protein_gym_name}.fasta"
    structure_path = base / "structure_sequence" / args.structure_vocab_size / f"{mapping.protein_gym_name}.fasta"
    if not residue_path.exists():
        raise FileNotFoundError(residue_path)
    if not structure_path.exists():
        raise FileNotFoundError(structure_path)
    pg_seq = read_fasta_sequence(residue_path)
    pg_structure = read_structure_tokens(structure_path)
    if len(pg_seq) != len(pg_structure):
        raise ValueError(f"{mapping.protein_gym_name} AA length {len(pg_seq)} != structure length {len(pg_structure)}")

    wt = WT_SEQUENCES[task]
    if mapping.offset + mapping.covered_length > len(pg_structure):
        raise ValueError(f"Mapping for {task} exceeds structure token length")
    if mapping.covered_length > len(wt):
        raise ValueError(f"Mapping for {task} exceeds ProSpero WT length")
    covered_wt = wt[: mapping.covered_length]
    structure_tokens = pg_structure[mapping.offset : mapping.offset + mapping.covered_length]
    return mapping, covered_wt, structure_tokens


def score_task(args, task, scorer):
    out_dir = Path(args.output_dir)
    rows_path = out_dir / f"{task}.csv"
    fieldnames = [
        "task",
        "split",
        "sequence",
        "fitness",
        "n_mutations",
        "n_scored_mutations",
        "n_skipped_mutations",
        "n_ignored_unstructured_mutations",
        "prosst_log_odds",
    ]
    completed = load_completed(rows_path)
    split_rows = collapsed_split(task, args.split)
    if args.max_sequences is not None:
        split_rows = split_rows[: args.max_sequences]
    pending = [(seq, fitness) for seq, fitness in split_rows if seq not in completed]
    mapping, covered_wt, structure_tokens = load_mapping_assets(args, task)
    full_wt = WT_SEQUENCES[task]
    covered_length = mapping.covered_length

    t0 = time.perf_counter()
    wt_log_probs = scorer.wt_position_log_probs(covered_wt, structure_tokens) if pending else None
    handle, writer = make_writer(rows_path, fieldnames)
    try:
        for idx, (seq, fitness) in enumerate(pending, start=1):
            score, scored, skipped, ignored = scorer.score_log_odds(seq, full_wt, wt_log_probs, covered_length)
            n_mut = len(mutation_positions(seq, full_wt, len(full_wt)))
            row = {
                "task": task,
                "split": args.split,
                "sequence": seq,
                "fitness": f"{fitness:.10g}",
                "n_mutations": n_mut,
                "n_scored_mutations": scored,
                "n_skipped_mutations": skipped,
                "n_ignored_unstructured_mutations": ignored,
                "prosst_log_odds": f"{score:.10g}",
            }
            writer.writerow(row)
            if idx % 50 == 0:
                handle.flush()
        handle.flush()
    finally:
        handle.close()
    seconds = time.perf_counter() - t0
    all_rows = list(load_completed(rows_path).values())
    summary = summarize(task, all_rows, out_dir, args.model_path, mapping, seconds=seconds)
    if summary:
        print(json.dumps(summary, sort_keys=True), flush=True)


def load_model(model_path, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval().to(device)
    return model, tokenizer


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model(args.model_path, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model_path,
                "seconds": time.perf_counter() - t0,
                "params": sum(p.numel() for p in model.parameters()),
            }
        ),
        flush=True,
    )
    scorer = ProSSTScorer(model, tokenizer, args.device)
    for task in args.tasks:
        print(f"Scoring {task}", flush=True)
        score_task(args, task, scorer)


if __name__ == "__main__":
    main()
