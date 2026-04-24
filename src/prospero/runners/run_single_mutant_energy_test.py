from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.runners.run_protein import (
    FROZEN_ESM_SURROGATE_ARCHS,
    get_parser as get_base_parser,
)
from prospero.surrogate import (
    Ensemble,
    build_surrogate_model,
    normalize_sequences,
    prepare_shared_esm_components,
)
from prospero.utils import set_seed

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class SingleMutantRecord:
    mutant: str
    position: int
    wt_residue: str
    mutant_residue: str
    predicted_fitness: float
    oracle_fitness: float
    predicted_delta_fitness: float
    oracle_delta_fitness: float
    predicted_delta_energy: float
    oracle_delta_energy: float


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _prepare_proxy(args, sequence_length: int, dataset: RegressionDataset) -> Ensemble:
    if getattr(args, "disable_esm_cache", False):
        args.cache_allowed_sequences = set()
        args.cache_allowed_sequences_ordered = []
        args.dataset_cache_task = None
    else:
        ordered = _dedupe_preserving_order(
            normalize_sequences(list(dataset.train) + list(dataset.valid))
        )
        args.cache_allowed_sequences = set(ordered)
        args.cache_allowed_sequences_ordered = ordered
        args.dataset_cache_task = args.task

    shared_esm_components = None
    if args.surrogate_arch in FROZEN_ESM_SURROGATE_ARCHS:
        shared_esm_components = prepare_shared_esm_components(args)

    proxy = Ensemble(
        [
            build_surrogate_model(
                sequence_length,
                args,
                shared_esm_components=shared_esm_components,
            )
            for _ in range(args.ensemble_size)
        ]
    )
    proxy.train(dataset)
    return proxy, shared_esm_components


def _release_proxy(proxy: Ensemble | None, shared_esm_components: object | None) -> None:
    del proxy
    if (
        shared_esm_components is not None
        and hasattr(shared_esm_components, "esm_batch_worker")
        and getattr(shared_esm_components, "esm_batch_worker") is not None
    ):
        try:
            shared_esm_components.esm_batch_worker.close()
        except Exception:
            pass
    del shared_esm_components
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _enumerate_single_mutants(wt_sequence: str) -> tuple[list[str], list[int], list[str]]:
    mutants: list[str] = []
    positions: list[int] = []
    mutant_residues: list[str] = []
    for idx, wt_residue in enumerate(wt_sequence):
        for aa in AA_ALPHABET:
            if aa == wt_residue:
                continue
            seq_chars = list(wt_sequence)
            seq_chars[idx] = aa
            mutants.append("".join(seq_chars))
            positions.append(idx)
            mutant_residues.append(aa)
    return mutants, positions, mutant_residues


def _fitness(oracle, task: str, sequences: list[str]) -> np.ndarray:
    if task.startswith("D_SHIFT"):
        values = oracle.get_fitness(sequences)
    else:
        values = oracle.get_fitness(np.asarray(sequences))
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _load_oracle_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_oracle_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get_oracle_scores_with_cache(
    *,
    task: str,
    wt_sequence: str,
    mutants: list[str],
    cache_path: Path | None,
) -> tuple[np.ndarray, float, bool]:
    if cache_path is not None:
        cache_payload = _load_oracle_cache(cache_path)
        task_payload = cache_payload.get(task)
        if (
            isinstance(task_payload, dict)
            and task_payload.get("wt_sequence") == wt_sequence
            and task_payload.get("mutants") == mutants
            and isinstance(task_payload.get("oracle_fitness"), list)
            and len(task_payload.get("oracle_fitness")) == len(mutants)
            and isinstance(task_payload.get("oracle_fitness_wt"), (int, float))
        ):
            return (
                np.asarray(task_payload["oracle_fitness"], dtype=np.float32),
                float(task_payload["oracle_fitness_wt"]),
                True,
            )

    oracle = get_landscape(task)
    oracle_fitness = _fitness(oracle, task, mutants)
    oracle_fitness_wt = float(_fitness(oracle, task, [wt_sequence])[0])

    if cache_path is not None:
        cache_payload = _load_oracle_cache(cache_path)
        cache_payload[task] = {
            "wt_sequence": wt_sequence,
            "num_single_mutants": int(len(mutants)),
            "mutants": list(mutants),
            "oracle_fitness": oracle_fitness.astype(np.float32).tolist(),
            "oracle_fitness_wt": oracle_fitness_wt,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_oracle_cache(cache_path, cache_payload)

    return oracle_fitness, oracle_fitness_wt, False


def _task_run(args, task: str) -> dict[str, object]:
    task_start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    wt_sequence = WT_SEQUENCES[task]
    dataset = RegressionDataset(task)
    timings["dataset_load_seconds"] = time.perf_counter() - t0

    args.task = task
    t0 = time.perf_counter()
    proxy, shared_esm_components = _prepare_proxy(args, len(wt_sequence), dataset)
    timings["surrogate_train_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mutants, positions, mutant_residues = _enumerate_single_mutants(wt_sequence)
    timings["single_mutant_enumeration_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    predicted_fitness = proxy.get_scores(mutants).detach().cpu().numpy().astype(np.float32)
    predicted_fitness_wt = float(proxy.get_scores([wt_sequence]).detach().cpu().item())
    timings["predicted_scoring_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _release_proxy(proxy, shared_esm_components)
    timings["surrogate_release_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    oracle_cache_path = (
        Path(args.oracle_cache_json)
        if getattr(args, "oracle_cache_json", None)
        else None
    )
    oracle_fitness, oracle_fitness_wt, oracle_cache_hit = _get_oracle_scores_with_cache(
        task=task,
        wt_sequence=wt_sequence,
        mutants=mutants,
        cache_path=oracle_cache_path,
    )
    timings["oracle_scoring_seconds"] = time.perf_counter() - t0
    timings["oracle_cache_hit"] = bool(oracle_cache_hit)

    predicted_delta_fitness = predicted_fitness - predicted_fitness_wt
    oracle_delta_fitness = oracle_fitness - oracle_fitness_wt

    predicted_delta_energy = -predicted_delta_fitness
    oracle_delta_energy = -oracle_delta_fitness

    best_pred_energy_idx = int(np.argmin(predicted_delta_energy))
    best_oracle_idx = int(np.argmax(oracle_delta_fitness))

    records: list[SingleMutantRecord] = []
    for i, mutant in enumerate(mutants):
        pos = int(positions[i])
        records.append(
            SingleMutantRecord(
                mutant=mutant,
                position=pos,
                wt_residue=wt_sequence[pos],
                mutant_residue=mutant_residues[i],
                predicted_fitness=float(predicted_fitness[i]),
                oracle_fitness=float(oracle_fitness[i]),
                predicted_delta_fitness=float(predicted_delta_fitness[i]),
                oracle_delta_fitness=float(oracle_delta_fitness[i]),
                predicted_delta_energy=float(predicted_delta_energy[i]),
                oracle_delta_energy=float(oracle_delta_energy[i]),
            )
        )

    predicted_rank_of_oracle_best = int(
        np.where(np.argsort(predicted_delta_energy) == best_oracle_idx)[0][0] + 1
    )
    oracle_rank_of_pred_best = int(
        np.where(np.argsort(oracle_delta_fitness)[::-1] == best_pred_energy_idx)[0][0] + 1
    )

    timings["task_total_seconds"] = time.perf_counter() - task_start
    return {
        "task": task,
        "wt_sequence_length": int(len(wt_sequence)),
        "num_single_mutants": int(len(mutants)),
        "energy_definition": "energy = -fitness",
        "wt_scores": {
            "predicted_fitness": predicted_fitness_wt,
            "oracle_fitness": oracle_fitness_wt,
        },
        "best_predicted_delta_energy": asdict(records[best_pred_energy_idx]),
        "best_oracle_improvement": asdict(records[best_oracle_idx]),
        "agreement": {
            "predicted_rank_of_oracle_best": predicted_rank_of_oracle_best,
            "oracle_rank_of_predicted_best": oracle_rank_of_pred_best,
            "pearson_predicted_vs_oracle_delta_fitness": float(
                np.corrcoef(predicted_delta_fitness, oracle_delta_fitness)[0, 1]
            ),
        },
        "per_mutant_rows": [asdict(record) for record in records],
        "timings_seconds": timings,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.description = (
        "Evaluate all single mutants for requested tasks, comparing predicted "
        "delta energy and oracle improvement."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["AAV", "LGK"],
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/0423_experiments/single_mutant_energy_test_aav_lgk.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-json-full",
        type=str,
        default="outputs/0423_experiments/single_mutant_energy_test_aav_lgk_full.json",
        help="Output path for full per-mutant JSON with LLM context definitions.",
    )
    parser.add_argument(
        "--oracle-cache-json",
        type=str,
        default=None,
        help=(
            "Optional shared cache JSON for oracle single-mutant scores. "
            "If provided, oracle scores are computed once and reused."
        ),
    )
    parser.set_defaults(
        surrogate_arch="interplm_mean_pool_ridge",
        ensemble_size=1,
        task="AAV",
        results_dirpath="outputs/0423_experiments",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    run_started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    set_seed(args.seed, args.full_deterministic)

    tasks = [task for task in args.tasks]
    task_results = [_task_run(args, task) for task in tasks]

    llm_context = {
        "purpose": (
            "Compare surrogate-predicted vs oracle-evaluated effects for all "
            "single mutants in each task."
        ),
        "definitions": {
            "fitness": "Higher is better according to the model.",
            "energy": "Defined here as energy = -fitness.",
            "delta_fitness": "fitness(mutant) - fitness(WT).",
            "delta_energy": "energy(mutant) - energy(WT) = -delta_fitness.",
            "best_predicted_delta_energy": (
                "Mutant with minimum predicted_delta_energy "
                "(largest predicted fitness gain)."
            ),
            "best_oracle_improvement": (
                "Mutant with maximum oracle_delta_fitness "
                "(largest true fitness gain)."
            ),
            "predicted_rank_of_oracle_best": (
                "1-based rank of the oracle-best mutant when sorting by "
                "predicted_delta_energy ascending."
            ),
            "oracle_rank_of_predicted_best": (
                "1-based rank of the predicted-best mutant when sorting by "
                "oracle_delta_fitness descending."
            ),
        },
        "columns_per_mutant_rows": {
            "mutant": "Mutant full sequence.",
            "position": "0-based mutation position in sequence.",
            "wt_residue": "Wild-type amino acid at position.",
            "mutant_residue": "Mutated amino acid at position.",
            "predicted_fitness": "Surrogate predicted fitness for mutant.",
            "oracle_fitness": "Oracle evaluated fitness for mutant.",
            "predicted_delta_fitness": "predicted_fitness - predicted_fitness_wt.",
            "oracle_delta_fitness": "oracle_fitness - oracle_fitness_wt.",
            "predicted_delta_energy": "-predicted_delta_fitness.",
            "oracle_delta_energy": "-oracle_delta_fitness.",
        },
        "notes": [
            "Each row is a single substitution mutant (Hamming distance 1 from WT).",
            "Amino-acid alphabet used: ACDEFGHIKLMNPQRSTVWY.",
            "Ranks are task-local and computed over all single mutants for that task.",
        ],
    }

    output = {
        "run_started_at_utc": run_started,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "surrogate_arch": str(args.surrogate_arch),
        "tasks": tasks,
        "total_runtime_seconds": float(time.perf_counter() - start),
        "results": task_results,
    }

    compact_results = []
    for task_result in task_results:
        compact_task_result = dict(task_result)
        compact_task_result.pop("per_mutant_rows", None)
        compact_results.append(compact_task_result)

    compact_output = dict(output)
    compact_output["results"] = compact_results

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(compact_output, handle, indent=2)

    full_output = dict(output)
    full_output["llm_context"] = llm_context
    full_output["format_version"] = "1.0"
    full_output["generated_by"] = "propero.runners.run_single_mutant_energy_test"
    full_output_path = Path(args.output_json_full)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    with full_output_path.open("w", encoding="utf-8") as handle:
        json.dump(full_output, handle, indent=2)

    print(f"saved: {output_path}")
    print(f"saved: {full_output_path}")


if __name__ == "__main__":
    main()
