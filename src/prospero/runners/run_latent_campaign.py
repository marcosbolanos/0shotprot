from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from prospero.latent_masked_search import run_latent_masked_single_iter


def _to_float(value: Any) -> float:
    return float(value)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overnight latent steering campaign runner",
    )
    parser.add_argument("--results-dirpath", type=str, default="outputs_latent_campaign")
    parser.add_argument("--task", type=str, default="LGK")
    parser.add_argument("--phase", type=str, default="phase0_baseline")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=20)

    parser.add_argument("--surrogate-arch", type=str, default="interplm_mean_pool_ridge")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--steering-layer", type=int, default=2)
    parser.add_argument(
        "--steering-direction-mode",
        type=str,
        default="signed",
        choices=["signed", "both", "positive", "negative"],
    )
    parser.add_argument(
        "--steering-scalars",
        type=float,
        nargs="+",
        default=[0.2, 0.7, 1.2, 1.8],
    )
    parser.add_argument("--combo-chunk-size", type=int, default=4)
    parser.add_argument("--mutation-penalty-lambda", type=float, default=0.0)
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument("--max-masked-mean-ppl", type=float, default=None)
    parser.add_argument("--max-masked-token-ppl", type=float, default=None)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument("--n-checks-multiplier", type=int, default=16)
    parser.add_argument("--kappa-scan", type=float, default=1.0)
    parser.add_argument("--kappa-guidance", type=float, default=0.1)

    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--proxy-batch-size", type=int, default=256)
    parser.add_argument("--num-model-max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs-per-valid", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--esm-model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--interplm-layer", type=int, default=2)
    parser.add_argument("--interplm-repo-id", type=str, default="Elana/InterPLM-esm2-8m")
    parser.add_argument("--interplm-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)

    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument("--full-deterministic", action="store_true", default=False)
    return parser


def _build_run_args(base: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        results_dirpath=base.results_dirpath,
        task=base.task,
        phase=base.phase,
        seed=int(seed),
        full_deterministic=bool(base.full_deterministic),
        n_iters=1,
        n_queries=base.top_k,
        surrogate_arch=base.surrogate_arch,
        batch_size=base.batch_size,
        top_k=base.top_k,
        top_features=base.top_features,
        steering_layer=base.steering_layer,
        steering_direction_mode=base.steering_direction_mode,
        steering_scalars=list(base.steering_scalars),
        combo_chunk_size=base.combo_chunk_size,
        mutation_penalty_lambda=base.mutation_penalty_lambda,
        max_mutations=base.max_mutations,
        max_masked_mean_ppl=base.max_masked_mean_ppl,
        max_masked_token_ppl=base.max_masked_token_ppl,
        min_corruptions=base.min_corruptions,
        max_corruptions=base.max_corruptions,
        n_checks_multiplier=base.n_checks_multiplier,
        kappa_scan=base.kappa_scan,
        kappa_guidance=base.kappa_guidance,
        ensemble_size=base.ensemble_size,
        proxy_batch_size=base.proxy_batch_size,
        num_model_max_epochs=base.num_model_max_epochs,
        patience=base.patience,
        epochs_per_valid=base.epochs_per_valid,
        lr=base.lr,
        weight_decay=base.weight_decay,
        ridge_alpha=base.ridge_alpha,
        ridge_fit_intercept=base.ridge_fit_intercept,
        esm_model_name=base.esm_model_name,
        esm_max_length=base.esm_max_length,
        interplm_layer=base.interplm_layer,
        interplm_repo_id=base.interplm_repo_id,
        interplm_normalized=base.interplm_normalized,
        sae_token_chunk_size=base.sae_token_chunk_size,
        alphabet=base.alphabet,
        disable_esm_cache=base.disable_esm_cache,
        debug_events=False,
        debug_heartbeat_seconds=15.0,
    )


def _build_fallbacks(batch_size: int, combo_chunk_size: int) -> list[tuple[int, int]]:
    pairs = [
        (batch_size, combo_chunk_size),
        (max(32, batch_size // 2), max(1, combo_chunk_size // 2)),
        (max(32, batch_size // 4), 1),
        (max(16, batch_size // 8), 1),
    ]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def main() -> None:
    args = get_parser().parse_args()
    seeds = list(range(args.seed_start, args.seed_end + 1))

    campaign_dir = Path(args.results_dirpath) / args.task / args.phase
    campaign_dir.mkdir(parents=True, exist_ok=True)
    summary_path = campaign_dir / "campaign_summary.json"
    rows_path = campaign_dir / "runs.jsonl"

    rows: list[dict[str, object]] = []
    for seed in seeds:
        fallback_pairs = _build_fallbacks(args.batch_size, args.combo_chunk_size)
        row: dict[str, object] = {
            "seed": seed,
            "status": "error",
            "error": "not_run",
        }
        for attempt_idx, (batch_size, combo_chunk_size) in enumerate(
            fallback_pairs, start=1
        ):
            run_args = _build_run_args(args, seed)
            run_args.batch_size = int(batch_size)
            run_args.combo_chunk_size = int(combo_chunk_size)
            try:
                run_json_path = run_latent_masked_single_iter(run_args)
                payload = json.loads(Path(run_json_path).read_text())
                row = {
                    "seed": seed,
                    "status": "ok",
                    "attempt": attempt_idx,
                    "used_batch_size": int(batch_size),
                    "used_combo_chunk_size": int(combo_chunk_size),
                    "run_json_path": str(run_json_path),
                    "best_oracle": payload["best_oracle"],
                    "oracle_top10_mean": payload["oracle_top10_mean"],
                    "surrogate_oracle_corr": payload["surrogate_oracle_corr"],
                    "num_generated": payload["num_generated"],
                    "num_unique": payload["num_unique"],
                    "mean_mutation_count": payload["mean_mutation_count"],
                    "best_sequence": payload["best_sequence"],
                    "mutated_positions": payload["mutated_positions"],
                    "total_runtime_seconds": payload["timings_seconds"]["total_runtime_seconds"],
                }
                break
            except Exception as exc:  # noqa: BLE001
                msg = repr(exc)
                row = {
                    "seed": seed,
                    "status": "error",
                    "attempt": attempt_idx,
                    "used_batch_size": int(batch_size),
                    "used_combo_chunk_size": int(combo_chunk_size),
                    "error": msg,
                }
                if "out of memory" in msg.lower():
                    continue
                break
        rows.append(row)
        with rows_path.open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item))
                handle.write("\n")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        best_oracles = np.asarray([_to_float(r["best_oracle"]) for r in ok_rows])
        top10 = np.asarray([_to_float(r["oracle_top10_mean"]) for r in ok_rows])
        diversity = np.asarray([
            _to_float(r["num_unique"]) / max(1.0, _to_float(r["num_generated"]))
            for r in ok_rows
        ])
        runtime = np.asarray([_to_float(r["total_runtime_seconds"]) for r in ok_rows])
        summary = {
            "task": args.task,
            "phase": args.phase,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "num_runs": len(rows),
            "num_ok": len(ok_rows),
            "num_error": len(rows) - len(ok_rows),
            "best_oracle_max": float(best_oracles.max()),
            "best_oracle_mean": float(best_oracles.mean()),
            "oracle_top10_mean_mean": float(top10.mean()),
            "diversity_mean": float(diversity.mean()),
            "runtime_mean_seconds": float(runtime.mean()),
            "runtime_total_seconds": float(runtime.sum()),
            "config": {
                "surrogate_arch": args.surrogate_arch,
                "steering_layer": args.steering_layer,
                "top_features": args.top_features,
                "steering_direction_mode": args.steering_direction_mode,
                "steering_scalars": list(args.steering_scalars),
                "mutation_penalty_lambda": args.mutation_penalty_lambda,
                "max_mutations": args.max_mutations,
                "max_masked_mean_ppl": args.max_masked_mean_ppl,
                "max_masked_token_ppl": args.max_masked_token_ppl,
                "batch_size": args.batch_size,
                "top_k": args.top_k,
            },
        }
    else:
        summary = {
            "task": args.task,
            "phase": args.phase,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "num_runs": len(rows),
            "num_ok": 0,
            "num_error": len(rows),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"campaign summary saved: {summary_path}")


if __name__ == "__main__":
    main()
