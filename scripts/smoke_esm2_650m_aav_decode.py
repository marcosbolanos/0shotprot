#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.landscapes import get_landscape

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def load_trace_masks(path: Path, limit: int, k: int) -> list[list[int]]:
    masks = []
    seen_particles = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "mask_selected":
                continue
            if event.get("optimization_round") not in {None, 1}:
                continue
            particle = event.get("particle")
            positions = event.get("mask_positions") or event.get("positions")
            key = (event.get("optimization_round"), event.get("generation_round"), particle)
            if key in seen_particles or not positions:
                continue
            pos = [int(p) for p in positions][:k]
            if len(pos) == k:
                masks.append(sorted(pos))
                seen_particles.add(key)
            if len(masks) >= limit:
                break
    if not masks:
        raise RuntimeError(f"No usable masks found in {path}")
    if len(masks) < limit:
        repeats = (limit + len(masks) - 1) // len(masks)
        masks = (masks * repeats)[:limit]
    return masks


def random_masks(sequence: str, limit: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    ids = list(range(len(sequence)))
    return [sorted(rng.sample(ids, k)) for _ in range(limit)]


def make_initial_tokens(batch_converter, alphabet, sequence: str, masks: list[list[int]], device: torch.device):
    data = [(f"seq_{i}", sequence) for i in range(len(masks))]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    for row, locs in enumerate(masks):
        for pos in locs:
            tokens[row, pos + 1] = alphabet.mask_idx  # ESM has BOS at position 0.
    return tokens


def untokenize_esm_tokens(tokens: torch.Tensor, alphabet) -> list[str]:
    seqs = []
    for row in tokens.detach().cpu().tolist():
        aas = []
        for tok in row[1:-1]:  # drop BOS/EOS
            aa = alphabet.get_tok(tok)
            if aa in AA20:
                aas.append(aa)
        seqs.append("".join(aas))
    return seqs


def decode_terminal(model, alphabet, tokens, masks, sequence, vocab_mode: str):
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA20}
    cluster = ALPHABETS["CHARGE"]
    batch_size = len(masks)
    k = len(masks[0])
    step_times = []
    total_log_delta = torch.zeros(batch_size, device=tokens.device)
    sampled = tokens.clone()

    with torch.inference_mode():
        for step in range(k):
            t0 = time.perf_counter()
            out = model(sampled, repr_layers=[], return_contacts=False)
            logits = out["logits"]
            if sampled.device.type == "cuda":
                torch.cuda.synchronize()
            forward_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            new_tokens = []
            for row, locs in enumerate(masks):
                pos = locs[step]
                original_aa = sequence[pos]
                if vocab_mode == "cluster":
                    allowed_aas = cluster[original_aa]
                elif vocab_mode == "full":
                    allowed_aas = AA20
                else:
                    raise ValueError(vocab_mode)
                allowed = torch.tensor([aa_to_idx[a] for a in allowed_aas], device=tokens.device)
                p = logits[row, pos + 1]
                log_probs = torch.log_softmax(p, dim=-1)
                probs = torch.softmax(p[allowed], dim=-1)
                pick_local = torch.multinomial(probs, 1).item()
                pick = allowed[pick_local]
                new_tokens.append(pick)
                total_log_delta[row] += log_probs[pick] - log_probs[aa_to_idx[original_aa]]
            new_tokens = torch.stack(new_tokens)
            for row, locs in enumerate(masks):
                sampled[row, locs[step] + 1] = new_tokens[row]
            if sampled.device.type == "cuda":
                torch.cuda.synchronize()
            sample_s = time.perf_counter() - t1
            step_times.append({"step": step + 1, "forward_s": forward_s, "sample_s": sample_s})
    return sampled, total_log_delta.detach().cpu().numpy(), step_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="AAV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mask_budget", type=int, default=4)
    parser.add_argument("--vocab", choices=["cluster", "full"], default="cluster")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace", type=Path, default=Path("outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603/n_samples_128/mixed_explore_exploit/AAV/debug_traces/seed_1.events.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/esm2_650m_smoke_aav_20260603/smoke_results.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sequence = WT_SEQUENCES[args.task]
    masks = load_trace_masks(args.trace, args.batch_size, args.mask_budget) if args.trace.exists() else random_masks(sequence, args.batch_size, args.mask_budget, args.seed)

    import esm

    t_load0 = time.perf_counter()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0
    n_params = sum(p.numel() for p in model.parameters())

    tokens = make_initial_tokens(batch_converter, alphabet, sequence, masks, device)

    # Warm up one forward on a tiny slice so setup kernels are not counted in decode speed.
    with torch.inference_mode():
        _ = model(tokens[: min(2, len(tokens))], repr_layers=[], return_contacts=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    sampled, zs_scores, step_times = decode_terminal(model, alphabet, tokens, masks, sequence, args.vocab)
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    seqs = untokenize_esm_tokens(sampled, alphabet)
    oracle = get_landscape(args.task)
    oracle_scores = oracle.get_fitness(seqs)

    result = {
        "task": args.task,
        "model": "esm2_t33_650M_UR50D",
        "params": n_params,
        "device": str(device),
        "batch_size": args.batch_size,
        "mask_budget": args.mask_budget,
        "vocab": args.vocab,
        "load_seconds": load_s,
        "decode_seconds": decode_s,
        "seconds_per_candidate": decode_s / args.batch_size,
        "seconds_per_forward_step_mean": float(np.mean([s["forward_s"] for s in step_times])),
        "step_times": step_times,
        "cuda_peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "oracle_best": float(np.max(oracle_scores)),
        "oracle_top100_mean": float(np.mean(np.sort(oracle_scores)[::-1][: min(100, len(oracle_scores))])),
        "oracle_mean": float(np.mean(oracle_scores)),
        "zero_shot_score_mean": float(np.mean(zs_scores)),
        "zero_shot_score_max": float(np.max(zs_scores)),
        "unique_sequences": int(len(set(seqs))),
        "examples": [
            {"sequence": seqs[i], "oracle": float(oracle_scores[i]), "zero_shot_score": float(zs_scores[i]), "mask": masks[i]}
            for i in range(min(5, len(seqs)))
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
