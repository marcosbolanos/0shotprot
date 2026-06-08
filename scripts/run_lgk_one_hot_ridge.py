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


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def normalize_sequences(sequences) -> list[str]:
    return ["".join(map(str, sequence)) for sequence in sequences]


def one_hot_features(sequences: list[str], alphabet: str = ALPHABET) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)
    seq_length = len(sequences[0])
    if any(len(sequence) != seq_length for sequence in sequences):
        raise ValueError("All sequences must have the same length for one-hot ridge")

    index_by_token = {token: idx for idx, token in enumerate(alphabet)}
    features = np.zeros((len(sequences), seq_length, len(alphabet)), dtype=np.float32)
    for row_idx, sequence in enumerate(sequences):
        for pos_idx, token in enumerate(sequence):
            try:
                features[row_idx, pos_idx, index_by_token[token]] = 1.0
            except KeyError as exc:
                raise ValueError(f"Unsupported amino-acid token {token!r}") from exc
    return features.reshape(len(sequences), -1)


def parse_top_k_values(raw_values: str) -> list[int]:
    values: list[int] = []
    for raw_value in raw_values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError("--top-k-values must contain positive integers")
        if value not in values:
            values.append(value)
    return values


def evaluate_model(
    *,
    name: str,
    train_features: np.ndarray,
    train_scores: np.ndarray,
    valid_features: np.ndarray,
    valid_scores: np.ndarray,
    ridge_alpha: float,
    fit_intercept: bool,
) -> dict[str, object]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features)
    x_valid = scaler.transform(valid_features)

    model = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept)
    model.fit(x_train, train_scores)
    valid_pred = model.predict(x_valid).astype(np.float32)
    train_pred = model.predict(x_train).astype(np.float32)
    spearman = spearmanr(valid_scores, valid_pred).correlation

    return {
        "name": name,
        "n_train": int(len(train_scores)),
        "train_r2": float(r2_score(train_scores, train_pred)),
        "valid_r2": float(r2_score(valid_scores, valid_pred)),
        "valid_rmse": float(np.sqrt(mean_squared_error(valid_scores, valid_pred))),
        "valid_mae": float(mean_absolute_error(valid_scores, valid_pred)),
        "valid_spearman": None if np.isnan(spearman) else float(spearman),
        "valid_predictions": valid_pred.tolist(),
    }


def write_predictions(
    *,
    path: Path,
    valid_sequences: list[str],
    valid_scores: np.ndarray,
    results: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_by_name = {
        str(result["name"]): result["valid_predictions"] for result in results
    }
    fieldnames = ["sequence", "oracle_score", *prediction_by_name.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sequence in enumerate(valid_sequences):
            row = {
                "sequence": sequence,
                "oracle_score": float(valid_scores[idx]),
            }
            for name, predictions in prediction_by_name.items():
                row[name] = float(predictions[idx])
            writer.writerow(row)


def write_plots(output_dir: Path, results: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        str(result["name"]).replace("_train", "").replace("full", "full train").replace("top", "top ")
        for result in results
    ]
    metrics = [
        ("valid_r2", "Validation R2"),
        ("valid_spearman", "Validation Spearman"),
        ("valid_rmse", "Validation RMSE"),
        ("valid_mae", "Validation MAE"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#264653", "#e76f51", "#f4a261", "#2a9d8f"]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = [float(result[metric]) for result in results]
        bars = ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="#333333", linewidth=0.8)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = 0.01 if value >= 0 else -0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=9,
            )
    fig.suptitle("LGK One-Hot Ridge Benchmarks", fontsize=16)
    fig.savefig(plot_dir / "lgk_one_hot_ridge_benchmark.png", dpi=180)
    plt.close(fig)

    subset_results = [result for result in results if result["name"] != "full_train"]
    xs = [int(result["n_train"]) for result in subset_results]
    full = next(result for result in results if result["name"] == "full_train")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(
        xs,
        [float(result["valid_spearman"]) for result in subset_results],
        marker="o",
        linewidth=2,
        color="#2a9d8f",
        label="Spearman",
    )
    ax.plot(
        xs,
        [float(result["valid_r2"]) for result in subset_results],
        marker="o",
        linewidth=2,
        color="#e76f51",
        label="R2",
    )
    ax.axhline(float(full["valid_spearman"]), color="#2a9d8f", linestyle="--", alpha=0.5, label="Full Spearman")
    ax.axhline(float(full["valid_r2"]), color="#e76f51", linestyle="--", alpha=0.5, label="Full R2")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Top-k training sequences")
    ax.set_ylabel("Validation metric")
    ax.set_title("LGK One-Hot Ridge vs Top-k Training Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(plot_dir / "lgk_one_hot_ridge_train_size.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGK ridge models on one-hot features.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--top-k-values",
        default="10",
        help="Comma-separated top-k training subset sizes to evaluate.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RegressionDataset("LGK")
    train_sequences = normalize_sequences(dataset.train)
    valid_sequences = normalize_sequences(dataset.valid)
    train_scores = np.asarray(dataset.train_scores, dtype=np.float32)
    valid_scores = np.asarray(dataset.valid_scores, dtype=np.float32)
    train_features = one_hot_features(train_sequences)
    valid_features = one_hot_features(valid_sequences)

    sorted_train_indices = np.argsort(train_scores)[::-1]
    top_k_values = parse_top_k_values(args.top_k_values)
    results = [
        evaluate_model(
            name="full_train",
            train_features=train_features,
            train_scores=train_scores,
            valid_features=valid_features,
            valid_scores=valid_scores,
            ridge_alpha=args.ridge_alpha,
            fit_intercept=args.ridge_fit_intercept,
        )
    ]
    top_k_train_scores: dict[str, list[float]] = {}
    for top_k in top_k_values:
        if top_k > len(train_scores):
            raise ValueError(
                f"Requested top-k={top_k}, but LGK train set has {len(train_scores)} rows"
            )
        top_k_indices = sorted_train_indices[:top_k]
        top_k_train_scores[f"top{top_k}_train"] = train_scores[top_k_indices].tolist()
        results.append(
            evaluate_model(
                name=f"top{top_k}_train",
                train_features=train_features[top_k_indices],
                train_scores=train_scores[top_k_indices],
                valid_features=valid_features,
                valid_scores=valid_scores,
                ridge_alpha=args.ridge_alpha,
                fit_intercept=args.ridge_fit_intercept,
            )
        )

    predictions_path = args.output_dir / "valid_predictions.csv"
    write_predictions(
        path=predictions_path,
        valid_sequences=valid_sequences,
        valid_scores=valid_scores,
        results=results,
    )

    for result in results:
        result.pop("valid_predictions", None)

    summary = {
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "LGK",
        "surrogate_arch": "one_hot_ridge",
        "representation": "flattened_positional_one_hot",
        "alphabet": ALPHABET,
        "n_train_full": int(len(train_scores)),
        "n_valid": int(len(valid_scores)),
        "top_k_values": top_k_values,
        "top_k_train_scores": top_k_train_scores,
        "results": results,
        "predictions_csv": str(predictions_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_plots(args.output_dir, results)
    print(f"saved: {summary_path}")
    print(f"saved: {predictions_path}")


if __name__ == "__main__":
    main()
