from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_sequences(sequences) -> list[str]:
    return ["".join(map(str, seq)) for seq in sequences]


def one_hot_features(sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 0), dtype=np.float32)
    length = len(sequences[0])
    aa_index = {aa: idx for idx, aa in enumerate(ALPHABET)}
    x = np.zeros((len(sequences), length * len(ALPHABET)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        if len(seq) != length:
            raise ValueError("All sequences for a task must have the same length.")
        for pos, aa in enumerate(seq):
            try:
                x[row, pos * len(ALPHABET) + aa_index[aa]] = 1.0
            except KeyError as exc:
                raise ValueError(f"Unsupported amino-acid token {aa!r}") from exc
    return x


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x)
    x_valid = scaler.transform(valid_x)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, train_y)
    return model.predict(x_train), model.predict(x_valid)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    sp = spearmanr(y_true, y_pred).correlation
    return {
        "spearman": None if np.isnan(sp) else float(sp),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ProSpero task learnability with pure one-hot ridge."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(WT_SEQUENCES), type=parse_csv_strings)
    parser.add_argument("--budgets", default="16,32,64,128,256,512", type=parse_csv_ints)
    parser.add_argument("--seeds", default="1,2,3,4,5", type=parse_csv_ints)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "surrogate_arch": "one_hot_ridge",
        "representation": "flattened_positional_one_hot",
        "alphabet": ALPHABET,
        "alpha": float(args.alpha),
        "budgets": args.budgets,
        "seeds": args.seeds,
        "tasks": {},
    }

    for task in args.tasks:
        dataset = RegressionDataset(task)
        train_sequences = normalize_sequences(dataset.train)
        valid_sequences = normalize_sequences(dataset.valid)
        train_y = np.asarray(dataset.train_scores, dtype=np.float32)
        valid_y = np.asarray(dataset.valid_scores, dtype=np.float32)
        train_x = one_hot_features(train_sequences)
        valid_x = one_hot_features(valid_sequences)
        task_records: list[dict[str, object]] = []

        for budget in args.budgets:
            if budget > len(train_y):
                continue
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                idx = np.sort(rng.choice(len(train_y), size=budget, replace=False))
                train_pred, valid_pred = fit_predict(
                    train_x[idx],
                    train_y[idx],
                    valid_x,
                    alpha=args.alpha,
                )
                train_metrics = evaluate(train_y[idx], train_pred)
                valid_metrics = evaluate(valid_y, valid_pred)
                rec = {
                    "task": task,
                    "budget": int(budget),
                    "seed": int(seed),
                    "n_train_full": int(len(train_y)),
                    "n_valid": int(len(valid_y)),
                    "sequence_length": int(len(train_sequences[0])),
                    "train_spearman": train_metrics["spearman"],
                    "valid_spearman": valid_metrics["spearman"],
                    "valid_r2": valid_metrics["r2"],
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_mae": valid_metrics["mae"],
                }
                records.append(rec)
                task_records.append(rec)

        # Full-train fit, deterministic pseudo-seed 0.
        train_pred, valid_pred = fit_predict(train_x, train_y, valid_x, alpha=args.alpha)
        train_metrics = evaluate(train_y, train_pred)
        valid_metrics = evaluate(valid_y, valid_pred)
        full_rec = {
            "task": task,
            "budget": "full",
            "seed": 0,
            "n_train_full": int(len(train_y)),
            "n_valid": int(len(valid_y)),
            "sequence_length": int(len(train_sequences[0])),
            "train_spearman": train_metrics["spearman"],
            "valid_spearman": valid_metrics["spearman"],
            "valid_r2": valid_metrics["r2"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
        }
        records.append(full_rec)
        task_records.append(full_rec)
        summary["tasks"][task] = summarize_task(task_records)

    with (args.output_dir / "one_hot_ridge_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {args.output_dir / 'summary.json'}")
    print(f"saved: {args.output_dir / 'one_hot_ridge_records.csv'}")


def summarize_task(records: list[dict[str, object]]) -> dict[str, object]:
    by_budget: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_budget.setdefault(str(rec["budget"]), []).append(rec)
    out = {
        "n_train_full": int(records[0]["n_train_full"]),
        "n_valid": int(records[0]["n_valid"]),
        "sequence_length": int(records[0]["sequence_length"]),
        "budgets": {},
    }
    for budget, budget_records in by_budget.items():
        vals = np.array([float(r["valid_spearman"]) for r in budget_records], dtype=float)
        out["budgets"][budget] = {
            "n_runs": len(budget_records),
            "valid_spearman_mean": float(np.mean(vals)),
            "valid_spearman_std": float(np.std(vals)),
            "valid_spearman_values": vals.tolist(),
            "valid_r2_mean": float(np.mean([float(r["valid_r2"]) for r in budget_records])),
            "valid_rmse_mean": float(np.mean([float(r["valid_rmse"]) for r in budget_records])),
        }
    return out


if __name__ == "__main__":
    main()
